# EcoAcoustic Sentinel

Desktop application for bird bioacoustic monitoring with a hybrid detection pipeline:
- Custom species library (few-shot embeddings)
- Global discovery with BirdNET

This repository is prepared for direct use from GitHub on Windows.

## Main Features

- Guided 4-step workflow in UI:
  1. Audio input/output
  2. Recording zone
  3. Detection parameters
  4. Export formats
- Hybrid detection strategy:
  - If custom profiles produce valid hits, those results are prioritized.
  - If no custom hit is found, BirdNET global discovery can be used.
- Species profile library:
  - Create/remove local species profiles from reference audio.
  - Multi-vector matching for higher precision.
- Results and monitoring:
  - Real-time progress and logs.
  - Summary tab with top species and chart.
  - Detailed detections table with playback support.
- Multiple export formats:
  - CSV
  - Raven Table
  - Audacity
  - Kaleidoscope

## Tech Stack

- UI: `PySide6`
- Bioacoustics/ML: `birdnet-analyzer`, `PyTorch`, `TensorFlow`, `torchaudio`
- Data processing: `numpy`, `pandas`, `scikit-learn`

## Requirements

- OS: Windows 10/11
- Python: **3.11.x** (required)
- Internet connection for first-time dependency/model setup

## Install and Run (Quick Start)

From PowerShell or CMD:

```bat
git clone https://github.com/Guivve-A/Eco_radar.git
cd Eco_radar
start_app.bat
```

`start_app.bat` automatically:
1. Creates/reuses `venv`
2. Runs environment setup
3. Executes a smoke test (`test_install.py`)
4. Launches the application (`main.py`)

## Manual Setup

```bat
setup_env.bat
call venv\Scripts\activate
python test_install.py
python main.py
```

## Optional GPU Setup

The default setup is CPU-first.
If you need a GPU build for PyTorch, adjust your environment and install from `requirements-gpu.txt` according to your CUDA setup.

## Typical Workflow

1. Select one audio file or a folder of audio files.
2. Select output folder.
3. Configure location mode:
   - Global mode (no location), or
   - Zone mode (lat/lon and optional metadata)
4. Tune parameters/preset.
5. Configure export formats.
6. Start analysis.
7. Review results in `Summary`, `Detections`, `Species Library`, and `Logs`.

## Output and Local Data

- `output/`: generated analysis results and exports
- `profiles/`: local species embeddings and profile metadata

The repository keeps folder structure with `.gitkeep` files; user-generated artifacts should stay local.

## Build Executable

### Build app (`onedir`)

```bat
python build_exe.py
```

### Build installer

1. Install Inno Setup 6
2. Run:

```bat
build_installer.bat
```

Expected installer output:
`dist\EcoAcousticSentinel_Installer_x64.exe`

## Troubleshooting

- Error: Python version not supported
  - Install Python 3.11.x and rerun `setup_env.bat`.
- Error during first install
  - Ensure internet access and rerun `setup_env.bat`.
- App opens but no detections
  - Check input path, supported audio formats, threshold settings, and zone/global mode.
- Local run does not change GitHub
  - Local execution affects only your machine unless you explicitly `git add`, `git commit`, and `git push`.

## Scientific Credit

BirdNET (Kahl et al., 2021):
BirdNET: A deep learning solution for avian diversity monitoring.
Ecological Informatics, 61, 101236.

## License

MIT. See `LICENSE`.
