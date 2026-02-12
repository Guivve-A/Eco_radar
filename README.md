# EcoAcoustic Sentinel (Online Ready)

Aplicacion de escritorio para monitoreo bioacustico aviar con arquitectura hibrida:
- Motor custom (Few-Shot por embeddings)
- Motor de descubrimiento global (BirdNET)

Este paquete esta preparado para publicarse en GitHub y ser inicializado por CMD.

## Requisitos

- Windows 10/11
- Python 3.11.x (obligatorio)
- Conexion a internet para la instalacion inicial

## Inicio rapido (1 comando)

Desde CMD o PowerShell en la raiz del proyecto:

```bat
start_app.bat
```

El script hace automaticamente:
1. crea/actualiza `venv` si no existe,
2. instala dependencias (`setup_env.bat`),
3. ejecuta `test_install.py`,
4. abre la aplicacion (`main.py`).

## Instalacion manual (si prefieres controlar cada paso)

```bat
setup_env.bat
call venv\Scripts\activate
python test_install.py
python main.py
```

## Publicacion en GitHub (maintainer)

```bat
git init
git add .
git commit -m "Initial release: EcoAcoustic Sentinel online"
```

Sugerencia:
- Mantener `profiles/.gitkeep` y `output/.gitkeep` para versionar estructura vacia.
- No subir `venv`, `dist`, `build`, ni resultados de usuario.

## Empaquetado ejecutable (.exe)

### Build app
```bat
python build_exe.py
```

### Build instalador
1. Instalar Inno Setup 6
2. Ejecutar:
```bat
build_installer.bat
```

El instalador final queda en `dist\EcoAcousticSentinel_Installer_x64.exe`.

## Arquitectura tecnica

- UI: PySide6
- Audio/ML: BirdNET + PyTorch/Torchaudio + TensorFlow
- Clustering: Scikit-Learn (KMeans/MiniBatchKMeans)
- Logica hibrida optimizada:
  - Si hay detecciones custom sobre umbral, se omite BirdNET global para ese archivo.
  - Si no hay hits custom, se activa descubrimiento global.

## Credito cientifico

BirdNET (Kahl et al., 2021):
BirdNET: A deep learning solution for avian diversity monitoring.
Ecological Informatics, 61, 101236.

## Licencia

MIT. Ver archivo LICENSE.
