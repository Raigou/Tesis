# Analizador de Movimiento Docente en el Aula

Herramienta de software desarrollada en Python que utiliza visión por computadora e inteligencia artificial para analizar el movimiento, la ubicación y el comportamiento de un docente a partir de la grabación de video de una clase. La aplicación procesa el video para extraer métricas cuantitativas y generar reportes visuales en PDF.

## Características Principales

- **Detección y Segmentación de Personas**: utiliza un modelo de la familia YOLOv11 (segmentación) para identificar y delinear la silueta de todas las personas presentes en el video.

- **Tracking del Docente**: una vez seleccionado manualmente en el primer fotograma, el sistema sigue al docente a lo largo del video mediante comparación de máscaras (IoU) entre fotogramas consecutivos.

- **Estimación de Profundidad**: implementa el modelo *Depth Anything* para calcular la distancia relativa del docente a la cámara, permitiendo un análisis espacial en pseudo-3D.

- **Análisis de Métricas Avanzadas**:
  - **Análisis Zonal**: porcentaje de tiempo que el docente pasa en cada una de las zonas del aula definidas por el usuario (o en 4 cuadrantes automáticos en modo no interactivo).
  - **Análisis de Proximidad**: distancia mínima promedio a otras personas y porcentaje de tiempo que el docente pasa cerca de alguien (umbral configurable).
  - **Análisis de Movimiento**: distancia total recorrida y porcentaje de tiempo quieto vs. en movimiento (umbral configurable).
  - **Análisis de Audio (opcional)**: si se proporciona una transcripción en CSV (con columnas `start`, `end`, `speaker`, `text`), calcula el Tiempo de Habla del Docente (*Teacher Talk Time*), el tiempo de habla de otros participantes, y el conteo de palabras.

- **Generación de Reportes Visuales**:
  - **Mapa de Calor de Posición**: muestra las zonas del aula donde el docente concentró su presencia, superpuesto sobre un fotograma del video.
  - **Gráfico de Trayectoria**: dibuja el recorrido del docente coloreado según la profundidad estimada, para visualizar su desplazamiento en el espacio.

- **Interfaz Gráfica (GUI)**: desarrollada con Tkinter, permite seleccionar video y transcripción, definir el intervalo de tiempo a analizar, ajustar el nivel de muestreo de fotogramas, configurar parámetros del modelo, y revisar el historial de ejecuciones por video.

- **Reportes en PDF**: compila todas las métricas y gráficos generados en un reporte PDF listo para compartir.

## Estructura del Proyecto

```
├── app_gui.py               # Interfaz gráfica (Tkinter): selección de video, control de ejecución, historial y PDF
├── main_opencv.py           # Script principal de procesamiento: orquesta el análisis, calcula métricas y guarda resultados
├── procesamiento_video.py   # Módulo con la lógica de visión por computadora (YOLO, profundidad, tracking, visualización)
├── config.ini                # Configuración de rutas, modelos y parámetros del análisis
└── requirements.txt          # Dependencias del proyecto
```

## Instalación

### 1. Prerrequisitos

- Python 3.9 o superior.
- Git instalado en tu sistema.
- (Opcional, recomendado) GPU NVIDIA con CUDA para acelerar la detección y la estimación de profundidad.

### 2. Clona el Repositorio

```bash
git clone https://github.com/Raigou/Tesis.git
cd Tesis
```

### 3. Crea y Activa un Entorno Virtual

```bash
# Crea el entorno virtual
python -m venv entorno

# Activa el entorno virtual
# En Windows:
.\entorno\Scripts\activate
# En macOS/Linux:
source entorno/bin/activate
```

### 4. Instala las Dependencias

```bash
pip install -r requirements.txt
```

> **Nota:** Si tienes una GPU NVIDIA compatible, instala la versión de PyTorch con soporte CUDA correspondiente a tu sistema ([ver guía oficial](https://pytorch.org/get-started/locally/)) para un rendimiento significativamente mayor. Sin GPU, la aplicación funciona igualmente sobre CPU, pero de forma más lenta.

### 5. Descarga el Modelo de IA

El modelo de detección (`yolo11l-seg.pt`) es un archivo grande y no se incluye en este repositorio.

1. Descárgalo desde: `[AQUÍ PEGAS EL ENLACE PÚBLICO DE DESCARGA]`.
2. Coloca el archivo en la carpeta principal del proyecto (la misma donde está `app_gui.py`), o bien en la ruta que indiques en `config.ini`.

El modelo de profundidad (*Depth Anything*) se descarga automáticamente desde Hugging Face la primera vez que se ejecuta el análisis.

## Configuración (`config.ini`)

Antes de la primera ejecución, revisa `config.ini`. También puedes editar estos valores desde la GUI, en el botón **⚙️ Configuración**.

| Sección | Parámetro | Descripción |
|---|---|---|
| `Paths` | `output_base_dir` | Carpeta donde se guardan los resultados de cada ejecución. |
| `Paths` | `default_video_dir` | Carpeta por defecto al buscar un video. |
| `Paths` | `default_transcript_dir` | Carpeta por defecto al buscar una transcripción CSV. |
| `Models` | `yolo_model` | Ruta o nombre del modelo YOLO de segmentación. |
| `Models` | `depth_model` | Nombre del modelo de profundidad (Hugging Face). |
| `Parameters` | `yolo_confidence` | Umbral de confianza para las detecciones de YOLO. |
| `Parameters` | `iou_threshold` | Umbral de IoU mínimo para mantener el tracking del docente entre fotogramas. |
| `Parameters` | `proximity_threshold_px` | Distancia (px) por debajo de la cual se considera que el docente está "cerca" de alguien. |
| `Parameters` | `stillness_threshold_px` | Desplazamiento (px) por debajo del cual se considera que el docente está "quieto". |

## Uso de la Aplicación

### 1. Ejecuta la Aplicación

Con el entorno virtual activado:

```bash
python app_gui.py
```

### 2. Configura el Análisis desde la GUI

1. **Seleccionar Video**: carga un archivo `.mp4`.
2. **Seleccionar CSV** *(opcional)*: carga una transcripción para habilitar el análisis de audio.
3. **Intervalo a Procesar**: ajusta los sliders de inicio y fin (MM:SS) para acotar el segmento del video a analizar.
4. **Nivel de Salto de Frames**: en 0 se analizan todos los fotogramas (más preciso, más lento); valores mayores saltan fotogramas para acelerar el procesamiento.
5. Haz clic en **🚀 Procesar Video**. Esto abre una ventana externa de OpenCV donde debes:
   - Hacer **clic sobre el docente** en el primer fotograma para seleccionarlo como sujeto de tracking.
   - Definir las **4 zonas del aula**: haz clic en la esquina superior-izquierda y luego en la esquina inferior-derecha de cada zona, una a la vez.
   - Una vez definidas las 4 zonas, el procesamiento comienza automáticamente. Puedes presionar `q` en cualquier momento para cancelar.

### 3. Revisa los Resultados

- Cada ejecución queda registrada en el **Historial**, identificada por fecha y hora. Haz doble clic sobre una fila para abrir su carpeta de resultados.
- Selecciona una ejecución del historial y usa **📊 Generar PDF** para compilar métricas y gráficos en un reporte, o **📄 Abrir PDF** si ya fue generado previamente.
- Cada carpeta de resultados (`<video>_Procesamiento_N (fecha_hora)`) contiene: `metricas.txt`, `heatmap_posicion.png`, `trayectoria.png` y, si fue generado, `reporte.pdf`.

## Notas

- El análisis de audio requiere que el CSV de transcripción tenga las columnas `start`, `end`, `speaker` y `text`, e identifica al docente con el valor de `speaker` igual a `SPEAKER_00`.
- Si `reportlab` o `Pillow` no están instalados, la generación y apertura de PDF quedará deshabilitada en la GUI, pero el resto del análisis funciona con normalidad.
