# Eco_radar
Simple bird identifier
UI Overview - EcoAcoustic Sentinel
Estructura general

    main.py:
        Ventana principal AudioAnalysisWindow.
        Flujo guiado por pasos en panel izquierdo:
            1. Audio
            2. Zona
            3. Parametros
            4. Exportacion
        Panel derecho de ejecucion y resultados:
            Progreso + metricas rapidas
            Tab Resumen (top especies + grafico)
            Tab Detecciones (tabla detallada + reproducir segmento)
            Tab Biblioteca de Especies (alta/baja de perfiles de referencia)
            Tab Logs (filtro por severidad, copiar, limpiar, colapsable)

    ui_theme.py:
        Paleta de colores (Palette)
        Tamanos/espaciados (Sizing)
        Estilos QSS centralizados (build_stylesheet())

    audio_analyzer.py:
        ProfileManager: guarda perfiles de referencia (vector centroid + metadata).
        AudioAnalyzer: motor few-shot por embeddings + similitud coseno.
        AudioWorker: ejecucion en hilo para no bloquear la UI.

Como agregar un nuevo parametro

    Agregar control en _build_step_params() dentro de main.py.
    Agregar tooltip y default:
        DEFAULTS
        PRESETS (si aplica)
    Guardar/restaurar en:
        _load_settings()
        closeEvent()
    Pasar el valor al comando BirdNET en _command_args().

Como agregar una nueva salida/export

    Agregar checkbox y mapeo en ExportOptionsDialog.
    Actualizar resumen visible en _refresh_export_summary().
    Incluir el formato en _command_args().
    Si requiere parser de resultados, extender:
        _load_results_from_output()
        _read_csv_rows() o agregar lector nuevo.

Flujo de perfiles de referencia

    Ir a Biblioteca de Especies.
    Click Agregar Nueva Especie.
    Ingresar nombre + seleccionar audio de referencia.
    El sistema extrae embeddings de BirdNET, calcula centroide y guarda el perfil.
    Durante analisis, cada chunk se compara contra estos perfiles con similitud coseno.

Notas UX

    Status chip en header: Listo, Analizando, Error.
    Validacion inmediata en Audio y Zona.
    Analisis no bloqueante con QProcess.
    Progreso real por parseo de lineas Files processed.

