# Analizador de Movimiento Docente en el Aula

Herramienta de software desarrollada en Python que utiliza visión por computadora e inteligencia artificial para analizar el movimiento, la ubicación y el comportamiento de un docente a partir de la grabación de video de una clase. La aplicación procesa el video para extraer métricas cuantitativas y generar reportes visuales en PDF.

## Características Principales

- **Detección y Segmentación de Personas**: utiliza un modelo de la familia YOLO11 (segmentación) para identificar y delinear la silueta de todas las personas presentes en el video.

- **Tracking del Docente**: una vez seleccionado manualmente en el primer fotograma, el sistema sigue al docente a lo largo del video mediante comparación de máscaras (IoU) entre fotogramas consecutivos.

- **Estimación de Profundidad**: implementa el modelo *Depth Anything* para calcular la distancia relativa del docente a la cámara, permitiendo un análisis espacial en pseudo-3D.

- **Análisis de Métricas Avanzadas**:
  - **Análisis Zonal**: porcentaje de tiempo que el docente pasa en cada una de las zonas del aula definidas por el usuario (o en 4 cuadrantes automáticos en modo no interactivo).
  - **Análisis de Proximidad**: distancia mínima promedio a otras personas y porcentaje de tiempo que el docente pasa cerca de alguien (umbral configurable).
  - **Análisis de Movimiento**: distancia total recorrida y porcentaje de tiempo quieto vs. en movimiento (umbral configurable).
  - **Análisis de Audio (opcional)**: si se proporciona una transcripción en CSV (con columnas `start`, `end`, `speaker`, `text`), calcula el Tiempo de Habla del Docente (*Teacher Talk Time*), el tiempo de habla de otros participantes y el conteo de palabras.

- **Generación de Reportes Visuales**:
  - **Mapa de Calor de Posición**: muestra las zonas del aula donde el docente concentró su presencia, superpuesto sobre un fotograma del video.
  - **Gráfico de Trayectoria**: dibuja el recorrido del docente coloreado según la profundidad estimada, para visualizar su desplazamiento en el espacio.

- **Interfaz Gráfica (GUI)**: desarrollada con Tkinter, permite seleccionar video y transcripción, definir el intervalo de tiempo a analizar, ajustar el nivel de muestreo de fotogramas, configurar parámetros del modelo y revisar el historial de ejecuciones por video.

- **Reportes en PDF**: compila todas las métricas y gráficos generados en un reporte PDF listo para compartir.

## Estructura del Proyecto

```
├── app_gui.py               # Interfaz gráfica (Tkinter): selección de video, control de ejecución, historial y PDF
├── main_opencv.py           # Script principal de procesamiento: orquesta el análisis, calcula métricas y guarda resultados
├── procesamiento_video.py   # Módulo con la lógica de visión por computadora (YOLO, profundidad, tracking, visualización)
├── config.ini               # Configuración de rutas, modelos y parámetros del análisis
└── requirements.txt         # Dependencias del proyecto
```

## Instalación

### 1. Prerrequisitos

- Python 3.9 o superior.
- Git instalado en tu sistema.
- Conexión a internet durante la primera ejecución (para descargar los modelos de IA).
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

> **Nota:** Si tienes una GPU NVIDIA compatible, instala la versión de PyTorch con soporte CUDA correspondiente a tu sistema ([guía oficial de PyTorch](https://pytorch.org/get-started/locally/)) para un rendimiento significativamente mayor. Sin GPU, la aplicación funciona igualmente sobre CPU, pero de forma más lenta.

### 5. Obtén los Modelos de IA

La aplicación usa dos modelos. Ninguno se incluye en el repositorio porque son archivos grandes, pero ambos son públicos y gratuitos.

#### Modelo de segmentación: YOLO11 (`yolo11l-seg.pt`)

Es un modelo oficial de [Ultralytics](https://docs.ultralytics.com/tasks/segment/). Tienes dos formas de obtenerlo:

- **Opción A, automática (recomendada):** no hagas nada. Si `yolo_model` en `config.ini` está en `yolo11l-seg.pt` y el archivo no existe localmente, la librería `ultralytics` lo descarga sola la primera vez que ejecutas el análisis y lo guarda en la carpeta del proyecto.
- **Opción B, manual (por ejemplo, si trabajas sin internet en el equipo final):**
  1. Descarga el archivo desde el repositorio oficial de pesos de Ultralytics: <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt>.
     Si el enlace directo cambia, entra a <https://github.com/ultralytics/assets/releases>, busca la sección *Assets* de la versión más reciente y descarga `yolo11l-seg.pt`.
  2. Copia el archivo a la carpeta principal del proyecto (la misma donde está `app_gui.py`), o a cualquier otra ruta y escribe esa ruta completa en `yolo_model` dentro de `config.ini`.

> **¿Equipo con poca capacidad?** Puedes usar una variante más liviana (`yolo11n-seg.pt`, `yolo11s-seg.pt` o `yolo11m-seg.pt`) descargándola del mismo lugar y cambiando `yolo_model` en `config.ini`. Será más rápida, con algo menos de precisión.

#### Modelo de profundidad: Depth Anything

Por defecto se usa [`LiheYoung/depth-anything-small-hf`](https://huggingface.co/LiheYoung/depth-anything-small-hf), indicado en `depth_model` dentro de `config.ini`. Se descarga automáticamente desde [Hugging Face](https://huggingface.co/) la primera vez que se ejecuta el análisis, sin que tengas que hacer nada. Los archivos quedan guardados en la caché local de Hugging Face (por defecto, `~/.cache/huggingface`), por lo que las ejecuciones siguientes funcionan sin internet.

## Configuración (`config.ini`)

El repositorio incluye un `config.ini` con valores por defecto que funcionan sin modificar nada. También puedes editar estos valores desde la GUI, en el botón **⚙️ Configuración**.

```ini
[Paths]
default_video_dir = ./videos
default_transcript_dir = ./transcripciones
output_base_dir = ./resultados

[Models]
yolo_model = yolo11l-seg.pt
depth_model = LiheYoung/depth-anything-small-hf

[Parameters]
yolo_confidence = 0.25
iou_threshold = 0.3
proximity_threshold_px = 150
stillness_threshold_px = 5
```

| Sección | Parámetro | Valor por defecto | Descripción |
|---|---|---|---|
| `Paths` | `default_video_dir` | `./videos` | Carpeta por defecto al buscar un video. |
| `Paths` | `default_transcript_dir` | `./transcripciones` | Carpeta por defecto al buscar una transcripción CSV. |
| `Paths` | `output_base_dir` | `./resultados` | Carpeta donde se guardan los resultados de cada ejecución. |
| `Models` | `yolo_model` | `yolo11l-seg.pt` | Nombre o ruta del modelo YOLO de segmentación. |
| `Models` | `depth_model` | `LiheYoung/depth-anything-small-hf` | Nombre del modelo de profundidad en Hugging Face. |
| `Parameters` | `yolo_confidence` | `0.25` | Umbral de confianza para las detecciones de YOLO. |
| `Parameters` | `iou_threshold` | `0.3` | Umbral de IoU mínimo para mantener el tracking del docente entre fotogramas. |
| `Parameters` | `proximity_threshold_px` | `150` | Distancia (px) por debajo de la cual se considera que el docente está "cerca" de alguien. |
| `Parameters` | `stillness_threshold_px` | `5` | Desplazamiento (px) por debajo del cual se considera que el docente está "quieto". |

**Sobre las rutas:** las rutas relativas (que empiezan con `./`) se interpretan desde la carpeta del proyecto, por lo que debes ejecutar `python app_gui.py` desde esa carpeta. Si lo prefieres, puedes reemplazarlas por rutas absolutas de tu equipo, por ejemplo `C:\Users\TuUsuario\Videos\clases` en Windows o `/home/tuusuario/videos` en Linux/macOS. Si una carpeta no existe, créala antes de usar la aplicación.

## Uso de la Aplicación

### 1. Ejecuta la Aplicación

Con el entorno virtual activado:

```bash
python app_gui.py
```

> La primera ejecución del análisis puede tardar más de lo normal, ya que en ese momento se descargan los modelos descritos en la sección de instalación.

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

## Formato de la Transcripción (CSV)

El análisis de audio requiere un CSV con las columnas `start`, `end`, `speaker` y `text`. Los tiempos `start` y `end` deben estar en segundos. El docente se identifica con el valor `SPEAKER_00` en la columna `speaker`; cualquier otro valor se considera otro participante.

Ejemplo:

```csv
start,end,speaker,text
0.0,5.2,SPEAKER_00,Buenos días a todos.
5.8,8.1,SPEAKER_01,Buenos días profesor.
```

## Solución de Problemas

- **Error al descargar un modelo en la primera ejecución:** verifica tu conexión a internet. Si estás detrás de un proxy o firewall, descarga `yolo11l-seg.pt` manualmente (Opción B de la instalación).
- **El análisis es muy lento:** usa una GPU NVIDIA con PyTorch compatible con CUDA, aumenta el *Nivel de Salto de Frames*, reduce el intervalo a procesar o cambia a una variante más liviana de YOLO11.
- **No aparece la opción de generar o abrir PDF:** instala las librerías faltantes con `pip install reportlab pillow`. Sin ellas, la generación de PDF queda deshabilitada, pero el resto del análisis funciona con normalidad.
