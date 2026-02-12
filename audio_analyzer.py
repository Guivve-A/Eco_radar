import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from PySide6.QtCore import QObject, Signal, Slot

import birdnet_analyzer.config as cfg
from birdnet_analyzer.analyze.utils import iterate_audio_chunks
from birdnet_analyzer.utils import ensure_model_exists


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma", ".aiff", ".aif"}


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


@dataclass
class ReferenceProfile:
    name: str
    safe_name: str
    source_audio: str
    vector_file: str
    created_at: str
    embedding_dim: int


class ProfileManager:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.storage_dir / "profiles.json"

    def _load_metadata(self) -> list[dict]:
        if not self.meta_path.exists():
            return []
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_metadata(self, items: list[dict]) -> None:
        self.meta_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    def _safe_name(self, name: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
        text = re.sub(r"_+", "_", text).strip("_")
        return text.lower() or "species"

    def list_profiles(self) -> list[ReferenceProfile]:
        return [ReferenceProfile(**item) for item in self._load_metadata()]

    def save_profile(self, name: str, source_audio: str, centroid: np.ndarray) -> ReferenceProfile:
        profiles = self._load_metadata()
        safe_name = self._safe_name(name)
        vector_file = f"{safe_name}.npy"
        vector_path = self.storage_dir / vector_file

        centroid = _l2_normalize(centroid.reshape(-1))
        np.save(vector_path, centroid.astype(np.float32))

        entry = ReferenceProfile(
            name=name.strip(),
            safe_name=safe_name,
            source_audio=str(source_audio),
            vector_file=vector_file,
            created_at=datetime.utcnow().isoformat(),
            embedding_dim=int(centroid.shape[0]),
        )

        updated = False
        for idx, item in enumerate(profiles):
            if item["name"].lower() == name.strip().lower() or item["safe_name"] == safe_name:
                profiles[idx] = asdict(entry)
                updated = True
                break
        if not updated:
            profiles.append(asdict(entry))
        self._save_metadata(profiles)
        return entry

    def remove_profile(self, profile_name: str) -> bool:
        profiles = self._load_metadata()
        kept: list[dict] = []
        removed = False

        for item in profiles:
            if item["name"].lower() == profile_name.strip().lower():
                vector_path = self.storage_dir / item["vector_file"]
                if vector_path.exists():
                    vector_path.unlink()
                removed = True
                continue
            kept.append(item)

        if removed:
            self._save_metadata(kept)
        return removed

    def load_profile_vectors(self) -> dict[str, np.ndarray]:
        vectors: dict[str, np.ndarray] = {}
        for profile in self.list_profiles():
            vector_path = self.storage_dir / profile.vector_file
            if not vector_path.exists():
                continue
            vec = np.load(vector_path).astype(np.float32).reshape(-1)
            vectors[profile.name] = _l2_normalize(vec)
        return vectors


class AudioAnalyzer:
    def __init__(
        self,
        profile_manager: ProfileManager,
        similarity_threshold: float = 0.75,
        overlap_seconds: float = 0.3,
        chunk_seconds: float = 3.0,
        batch_size: int = 1,
        threads: int = 4,
    ) -> None:
        self.profile_manager = profile_manager
        self.similarity_threshold = float(similarity_threshold)
        self.overlap_seconds = float(overlap_seconds)
        self.chunk_seconds = float(chunk_seconds)
        self.batch_size = int(batch_size)
        self.threads = int(threads)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cosine = torch.nn.CosineSimilarity(dim=1)
        ensure_model_exists()

    def _collect_audio_files(self, input_path: str) -> list[Path]:
        p = Path(input_path)
        if not p.exists():
            return []
        if p.is_file():
            return [p] if p.suffix.lower() in AUDIO_EXTENSIONS else []

        files: list[Path] = []
        for ext in AUDIO_EXTENSIONS:
            files.extend(p.rglob(f"*{ext}"))
        return sorted(files)

    def _configure_birdnet(self, audio_input: str) -> None:
        cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
        cfg.LABELS_FILE = cfg.BIRDNET_LABELS_FILE
        cfg.SAMPLE_RATE = cfg.BIRDNET_SAMPLE_RATE
        cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH

        cfg.INPUT_PATH = audio_input
        cfg.SIG_OVERLAP = max(0.0, min(2.9, float(self.overlap_seconds)))
        cfg.AUDIO_SPEED = 1.0
        cfg.BANDPASS_FMIN = 0
        cfg.BANDPASS_FMAX = 15000
        cfg.BATCH_SIZE = max(1, int(self.batch_size))

        input_obj = Path(audio_input)
        if input_obj.is_dir():
            cfg.CPU_THREADS = max(1, int(self.threads))
            cfg.TFLITE_THREADS = 1
        else:
            cfg.CPU_THREADS = 1
            cfg.TFLITE_THREADS = max(1, int(self.threads))

    def _estimate_total_segments(self, files: list[Path]) -> int:
        total = 0
        step = max(0.1, self.chunk_seconds - self.overlap_seconds)
        for file_path in files:
            try:
                duration = float(sf.info(str(file_path)).duration)
                segments = max(1, int(math.ceil(max(duration - self.chunk_seconds, 0.0) / step)) + 1)
            except Exception:
                segments = 1
            total += segments
        return max(total, 1)

    def _extract_file_embeddings(self, file_path: Path):
        self._configure_birdnet(str(file_path))
        for start, end, emb in iterate_audio_chunks(str(file_path), embeddings=True):
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            if np.linalg.norm(vec) <= 1e-12:
                continue
            yield float(start), float(end), _l2_normalize(vec)

    def build_reference_centroid(self, reference_audio: str) -> tuple[np.ndarray, int]:
        vectors: list[np.ndarray] = []
        for _, _, emb in self._extract_file_embeddings(Path(reference_audio)):
            vectors.append(emb)
        if not vectors:
            raise ValueError("No se pudieron extraer embeddings validos del audio de referencia.")

        matrix = np.vstack(vectors).astype(np.float32)
        centroid = _l2_normalize(matrix.mean(axis=0))
        return centroid, len(vectors)

    def add_reference_profile(self, profile_name: str, reference_audio: str) -> tuple[ReferenceProfile, int]:
        centroid, used_chunks = self.build_reference_centroid(reference_audio)
        profile = self.profile_manager.save_profile(profile_name, reference_audio, centroid)
        return profile, used_chunks

    def analyze(
        self,
        input_path: str,
        *,
        similarity_threshold: float | None = None,
        progress_callback: Callable[[int, int, str, int, float], None] | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[list[dict], dict]:
        threshold = float(similarity_threshold if similarity_threshold is not None else self.similarity_threshold)
        profile_vectors = self.profile_manager.load_profile_vectors()
        if not profile_vectors:
            raise ValueError("No hay perfiles de referencia cargados. Agrega especies en 'Biblioteca de Especies'.")

        files = self._collect_audio_files(input_path)
        if not files:
            raise ValueError("No hay archivos de audio validos para analizar.")

        names = list(profile_vectors.keys())
        matrix = np.vstack([profile_vectors[name] for name in names]).astype(np.float32)
        centroids = torch.from_numpy(matrix).to(self.device)

        total_segments = self._estimate_total_segments(files)
        processed = 0
        detections: list[dict] = []
        running_conf: list[float] = []

        for file_path in files:
            if log_callback:
                log_callback("Info", f"Analizando: {file_path.name}")

            for start, end, emb in self._extract_file_embeddings(file_path):
                if should_stop and should_stop():
                    if log_callback:
                        log_callback("Warning", "Analisis detenido por el usuario.")
                    summary = {
                        "processed_segments": processed,
                        "total_segments": total_segments,
                        "detections": len(detections),
                        "avg_confidence": float(mean(running_conf)) if running_conf else 0.0,
                        "stopped": True,
                    }
                    return detections, summary

                emb_t = torch.from_numpy(emb).to(self.device)
                scores = self.cosine(centroids, emb_t.unsqueeze(0))

                best_idx = int(torch.argmax(scores).item())
                best_score = float(scores[best_idx].item())
                processed += 1

                if best_score >= threshold:
                    matched_name = names[best_idx]
                    running_conf.append(best_score)
                    detections.append(
                        {
                            "start": start,
                            "end": end,
                            "scientific": matched_name,
                            "common": matched_name,
                            "confidence": best_score,
                            "file": str(file_path),
                            "source": "embedding_similarity",
                        }
                    )

                avg_conf = float(mean(running_conf)) if running_conf else 0.0
                if progress_callback:
                    progress_callback(processed, total_segments, str(file_path), len(detections), avg_conf)

        summary = {
            "processed_segments": processed,
            "total_segments": total_segments,
            "detections": len(detections),
            "avg_confidence": float(mean(running_conf)) if running_conf else 0.0,
            "stopped": False,
        }
        return detections, summary

    def export_detections(
        self,
        output_dir: str,
        detections: list[dict],
        formats: list[str],
        zone_metadata: dict | None = None,
    ) -> list[Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported: list[Path] = []

        if "csv" in formats:
            csv_path = out_dir / f"EmbeddingDetections_{timestamp}.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                header = ["Start (s)", "End (s)", "Scientific name", "Common name", "Confidence", "File", "Source"]
                if zone_metadata:
                    header.extend(["Country", "Province", "Canton", "Latitude", "Longitude", "RadiusKm"])
                writer.writerow(header)

                for row in detections:
                    line = [
                        f"{row['start']:.2f}",
                        f"{row['end']:.2f}",
                        row["scientific"],
                        row["common"],
                        f"{row['confidence']:.4f}",
                        row["file"],
                        row.get("source", "embedding_similarity"),
                    ]
                    if zone_metadata:
                        line.extend(
                            [
                                zone_metadata.get("country", ""),
                                zone_metadata.get("province", ""),
                                zone_metadata.get("canton", ""),
                                zone_metadata.get("lat", ""),
                                zone_metadata.get("lon", ""),
                                zone_metadata.get("radius_km", ""),
                            ]
                        )
                    writer.writerow(line)
            exported.append(csv_path)

        if "table" in formats:
            table_path = out_dir / f"EmbeddingDetections_{timestamp}.BirdNET.selection.table.txt"
            with table_path.open("w", encoding="utf-8") as f:
                f.write(
                    "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\t"
                    "Species Code\tCommon Name\tScientific Name\tConfidence\n"
                )
                for idx, row in enumerate(detections, start=1):
                    sp = row["common"]
                    code = re.sub(r"[^A-Z]", "", sp.upper())[:6] or "CUSTOM"
                    f.write(
                        f"{idx}\tSpectrogram 1\t1\t{row['start']:.2f}\t{row['end']:.2f}\t0\t15000\t{code}\t"
                        f"{row['common']}\t{row['scientific']}\t{row['confidence']:.4f}\n"
                    )
            exported.append(table_path)

        if "audacity" in formats:
            aud_path = out_dir / f"EmbeddingDetections_{timestamp}.Audacity.labels.txt"
            with aud_path.open("w", encoding="utf-8") as f:
                for row in detections:
                    f.write(f"{row['start']:.2f}\t{row['end']:.2f}\t{row['common']} ({row['confidence']:.2f})\n")
            exported.append(aud_path)

        if "kaleidoscope" in formats:
            kal_path = out_dir / f"EmbeddingDetections_{timestamp}.kaleidoscope.csv"
            with kal_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["FOLDER", "IN FILE", "OFFSET", "DURATION", "SCIENTIFIC NAME", "COMMON NAME", "CONFIDENCE"])
                for row in detections:
                    file_path = Path(row["file"])
                    writer.writerow(
                        [
                            str(file_path.parent),
                            file_path.name,
                            f"{row['start']:.2f}",
                            f"{max(0.1, row['end'] - row['start']):.2f}",
                            row["scientific"],
                            row["common"],
                            f"{row['confidence']:.4f}",
                        ]
                    )
            exported.append(kal_path)

        return exported


class AudioWorker(QObject):
    progress = Signal(int, int, str, int, float)
    log = Signal(str, str)
    finished = Signal(object, object, object)  # detections, summary, exported_files
    failed = Signal(str)

    def __init__(
        self,
        analyzer: AudioAnalyzer,
        input_path: str,
        output_dir: str,
        export_formats: list[str],
        *,
        zone_metadata: dict | None = None,
        similarity_threshold: float | None = None,
        overlap_seconds: float | None = None,
        threads: int | None = None,
    ) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.input_path = input_path
        self.output_dir = output_dir
        self.export_formats = export_formats
        self.zone_metadata = zone_metadata
        self.similarity_threshold = similarity_threshold
        self.overlap_seconds = overlap_seconds
        self.threads = threads
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return self._stop_requested

    @Slot()
    def run(self) -> None:
        try:
            if self.overlap_seconds is not None:
                self.analyzer.overlap_seconds = float(self.overlap_seconds)
            if self.threads is not None:
                self.analyzer.threads = int(self.threads)

            detections, summary = self.analyzer.analyze(
                self.input_path,
                similarity_threshold=self.similarity_threshold,
                progress_callback=lambda done, total, current_file, det_count, avg_conf: self.progress.emit(
                    done, total, current_file, det_count, avg_conf
                ),
                log_callback=lambda level, text: self.log.emit(level, text),
                should_stop=self._should_stop,
            )
            exported = self.analyzer.export_detections(
                self.output_dir,
                detections,
                self.export_formats,
                zone_metadata=self.zone_metadata,
            )
            self.finished.emit(detections, summary, [str(p) for p in exported])
        except Exception as ex:
            self.failed.emit(str(ex))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
