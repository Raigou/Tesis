Analizador de Movimiento Docente en el Aula
Este proyecto es una herramienta de software desarrollada en Python que utiliza visión por computadora e inteligencia artificial para analizar el movimiento y comportamiento de un docente a partir de una grabación de video de una clase. La aplicación procesa el video para extraer métricas detalladas y generar reportes visuales.

Características Principales
Detección y Segmentación de Personas: Utiliza un modelo de la familia YOLOv11 para identificar con precisión a todas las personas en el video, delineando sus siluetas.

Estimación de Profundidad: Implementa el modelo Depth Anything para calcular la distancia relativa de las personas a la cámara, permitiendo un análisis en 3D del espacio.

Análisis de Métricas Avanzadas: Calcula una variedad de métricas clave, incluyendo:

Análisis Zonal: Tiempo que el docente pasa en diferentes zonas predefinidas del aula.

Análisis de Proximidad: Distancia promedio a otras personas y tiempo que pasa cerca de ellas.

Análisis de Movimiento: Distancia total recorrida y porcentaje de tiempo quieto vs. en movimiento.

Generación de Reportes Visuales: Crea automáticamente dos tipos de gráficos para una fácil interpretación:

Mapa de Calor de Posición: Muestra las áreas del aula donde el docente concentró su presencia.

Gráfico de Trayectoria: Dibuja el recorrido del docente, coloreado por profundidad para visualizar su movimiento en el espacio.

Interfaz Gráfica de Usuario (GUI): Desarrollada con Tkinter para una operación sencilla e intuitiva, permitiendo seleccionar videos, definir intervalos de tiempo y configurar el análisis.

Reportes en PDF: Compila todas las métricas y gráficos en un reporte PDF profesional y fácil de compartir.

Instalación
Sigue estos pasos para configurar el entorno y ejecutar el proyecto.

1. Prerrequisitos
Python 3.9 o superior.

Git instalado en tu sistema.

2. Clona el Repositorio
Abre una terminal y clona este repositorio en tu computadora.

git clone https://github.com/Raigou/Tesis.git
cd Tesis

3. Crea y Activa un Entorno Virtual
Es una buena práctica trabajar en un entorno virtual para aislar las dependencias del proyecto.

# Crea el entorno virtual
python -m venv entorno

# Activa el entorno virtual
# En Windows:
.\entorno\Scripts\activate
# En macOS/Linux:
source entorno/bin/activate

4. Instala las Dependencias
Con el entorno virtual activado, instala todas las librerías necesarias ejecutando el siguiente comando:

pip install -r requirements.txt

Nota: Si tienes una GPU NVIDIA compatible, se recomienda instalar una versión de PyTorch con soporte para CUDA para un rendimiento significativamente mayor.

5. Descarga el Modelo de IA
El modelo de detección es un archivo grande y no se incluye directamente en este repositorio.

Descarga el archivo del modelo yolo11l-seg.pt desde [AQUÍ PEGAS EL ENLACE PÚBLICO DE DESCARGA DE TU GOOGLE DRIVE, MEGA, ETC.].

Una vez descargado, coloca el archivo yolo11l-seg.pt en la carpeta principal del proyecto (la misma carpeta donde está app_gui.py).

Uso de la Aplicación
Configura las Rutas (Opcional): Antes de la primera ejecución, puedes abrir el archivo config.ini con un editor de texto. Aquí puedes cambiar las rutas por defecto para tus carpetas de videos y la carpeta donde se guardarán los resultados.

Ejecuta la Aplicación:
Con el entorno virtual aún activado, ejecuta el siguiente comando en la terminal:

python app_gui.py

Opera desde la Interfaz:

Usa el botón "Seleccionar Video" para cargar un archivo .mp4.

Ajusta los controles deslizantes para definir el intervalo de tiempo que deseas analizar.

Usa el slider de "Velocidad vs. Precisión" para decidir si analizar todos los fotogramas (más preciso) o saltarse algunos (más rápido).

Haz clic en "Procesar Video" para iniciar el análisis.

Una vez finalizado, selecciona una ejecución del historial y usa los botones "Generar PDF" o "Abrir PDF" para ver los resultados.
