# main_opencv.py (CORREGIDO - Numeración de Historial)
import cv2
import numpy as np
import os
import torch
import procesamiento_video as proc
import time
import argparse
import sys
import datetime
import pandas as pd
import glob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import configparser

# --- Funcion para parsear argumentos ---
def parse_args():
    parser = argparse.ArgumentParser(description="Procesamiento de video para analisis de movimiento docente.")
    parser.add_argument('--video', required=True, help='Ruta al archivo de video MP4/AVI.')
    parser.add_argument('--start', type=float, default=0.0, help='Tiempo de inicio en segundos.')
    parser.add_argument('--end', type=float, default=-1.0, help='Tiempo de fin en segundos (-1 para final).')
    parser.add_argument('--transcript', default=None, help='(Opcional) Ruta al archivo CSV de transcripcion.')
    parser.add_argument('--frame-skip', type=int, default=0, help='Analizar 1 de cada N frames. 0 para analizar todos.')
    parser.add_argument('--non-interactive', action='store_true', help='Ejecutar en modo no interactivo.')
    return parser.parse_args()

# --- Funcion auxiliar para crear mascaras de zona ---
def crear_mascara_zona(coords, h, w):
    mask_uint8 = np.zeros((h, w), dtype=np.uint8); x1, y1, x2, y2 = map(int, coords); x1_safe, y1_safe = max(0, x1), max(0, y1); x2_safe, y2_safe = min(w, x2), min(h, y2)
    if x1_safe < x2_safe and y1_safe < y2_safe: cv2.rectangle(mask_uint8, (x1_safe, y1_safe), (x2_safe, y2_safe), 255, -1)
    mask_bool = (mask_uint8 > 0); return mask_bool

# --- Leer configuración desde config.ini ---
config = configparser.ConfigParser()
config.read('config.ini')

# --- Usar valores de config.ini ---
carpeta_destino_base = config.get('Paths', 'output_base_dir')
ruta_modelo_yolo = config.get('Models', 'yolo_model')
nombre_ventana_unica = "Procesamiento Tesis"
confianza_yolo = config.getfloat('Parameters', 'yolo_confidence')
iou_tracking_threshold = config.getfloat('Parameters', 'iou_threshold')
depth_model_name = config.get('Models', 'depth_model')
zone_names = ["Zona 1", "Zona 2", "Zona 3", "Zona 4"]; zones_to_define = len(zone_names)
threshold_cercania_px = config.getint('Parameters', 'proximity_threshold_px')
threshold_quietud_px = config.getint('Parameters', 'stillness_threshold_px')

# --- Parsear Argumentos y Cargar Transcript ---
args = parse_args(); ruta_video = args.video; start_time_sec = args.start; end_time_sec = args.end; ruta_transcript = args.transcript
frame_skip = args.frame_skip
if not os.path.exists(ruta_video): print(f"❌ Error: Video no existe: {ruta_video}"); sys.exit(1)
if start_time_sec < 0: start_time_sec = 0.0
if end_time_sec != -1.0 and end_time_sec <= start_time_sec: print(f"❌ Error: Tiempo final <= inicial."); sys.exit(1)
transcript_df = None; transcript_available = False
if ruta_transcript:
    print(f"🔄 Intentando cargar transcripcion desde: {ruta_transcript}")
    if not os.path.exists(ruta_transcript): print(f"⚠️ Advertencia: Archivo de transcripcion no encontrado.")
    else:
        try:
            transcript_df = pd.read_csv(ruta_transcript, index_col=0); required_cols = ['start', 'end', 'speaker', 'text']
            if all(col in transcript_df.columns for col in required_cols):
                transcript_available = True; print("✅ Transcripcion CSV cargada.")
                transcript_df = transcript_df.dropna(subset=required_cols); transcript_df['start'] = pd.to_numeric(transcript_df['start'], errors='coerce'); transcript_df['end'] = pd.to_numeric(transcript_df['end'], errors='coerce'); transcript_df = transcript_df.dropna(subset=['start', 'end'])
            else: print(f"❌ Error: CSV no contiene columnas: {required_cols}")
        except Exception as e: print(f"❌ Error al leer/procesar CSV: {e}")
else: print("ℹ️ No se proporciono archivo de transcripcion (--transcript).")

# Determinar dispositivo
if torch.cuda.is_available(): device_type = 'cuda'; print("✅ CUDA (GPU) disponible. Usando GPU.")
else: device_type = 'cpu'; print("⚠️ CUDA no disponible. Usando CPU.")

# --- Inicializacion ---
start_time_script = time.time(); nombre_video_base = os.path.splitext(os.path.basename(ruta_video))[0]

# --- INICIO DEL BLOQUE CORREGIDO ---
print("🔄 Buscando ejecuciones anteriores...")
next_run_number = 1
try:
    search_pattern = os.path.join(carpeta_destino_base, f"{nombre_video_base}_Procesamiento_*")
    existing_runs = glob.glob(search_pattern)
    
    if existing_runs:
        max_n = 0
        for folder in existing_runs:
            folder_name = os.path.basename(folder)
            match = re.search(r"_Procesamiento_(\d+)", folder_name)
            if match:
                num = int(match.group(1))
                if num > max_n:
                    max_n = num
        
        next_run_number = max_n + 1
        
    print(f"    Siguiente numero de ejecución: {next_run_number}")
    
except Exception as e:
    print(f"⚠️ Advertencia al determinar el número de ejecución: {e}")
    next_run_number = 1
# --- FIN DEL BLOQUE CORREGIDO ---

run_timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"); run_folder_name = f"{nombre_video_base}_Procesamiento_{next_run_number} ({run_timestamp_str})"
run_output_dir = os.path.join(carpeta_destino_base, run_folder_name)
try: os.makedirs(run_output_dir, exist_ok=True); print(f"➡️  Resultados en: {run_output_dir}")
except OSError as e: print(f"❌ Error creando carpeta '{run_output_dir}': {e}"); sys.exit(1)

modelo_yolo = proc.cargar_modelo_yolo(ruta_modelo_yolo); modelo_profundidad, procesador_profundidad = proc.cargar_modelo_profundidad(depth_model_name, device=device_type)
if modelo_yolo is None or modelo_profundidad is None: sys.exit(1)
cap = cv2.VideoCapture(ruta_video);
if not cap.isOpened(): print(f"❌ Error al abrir video: {ruta_video}"); sys.exit(1)
ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps_video_original = cap.get(cv2.CAP_PROP_FPS); video_total_duration_sec = -1.0
if fps_video_original > 0: frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
if frame_count_total > 0: video_total_duration_sec = frame_count_total / fps_video_original; print(f"Video Info: {ancho}x{alto} @ {fps_video_original:.2f} FPS (Total: {video_total_duration_sec:.2f}s)")
else: fps_video_original = 30; print("⚠️ FPS/Count invalido, usando 30 FPS.")
start_frame_num = int(start_time_sec * fps_video_original)
end_frame_num = int(end_time_sec * fps_video_original) if end_time_sec > 0 else frame_count_total
if frame_skip > 0:
    total_frames_to_process = (end_frame_num - start_frame_num) / (frame_skip + 1)
else:
    total_frames_to_process = end_frame_num - start_frame_num

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)
print(f"Video posicionado en frame {start_frame_num}.")
docente_mask_actual = None; tracking_activado = False; heatmap_posicion_acumulado = np.zeros((alto, ancho), dtype=np.float32)
results_primer_frame = None; teacher_path = []; frame_inicial = None
frames_procesados_post_seleccion = 0; frames_con_docente_trackeado = 0; total_personas_detectadas_acum = 0; lista_profundidades_docente = []
start_time_processing_loop = None; current_mode = 'SELECT_TEACHER'; current_zone_index = 0; first_corner = None
defined_zones = {}; last_click_coords = None; zone_frame_counts = {} ; indefinido_frame_count = 0; zone_masks = {}; zone_init_error = False
lista_distancias_minimas = []; frames_cercano_a_alguien = 0; lista_personas_cercanas_count = []
frame_read_counter = 0; last_processed_overlay = None
# --- Fase 1 & 2 (Interacción) ---
def on_mouse_click(event, x, y, flags, param):
    global last_click_coords, current_mode, first_corner, current_zone_index, defined_zones, tracking_activado, docente_mask_actual
    if args.non_interactive: return
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_mode == 'SELECT_TEACHER': last_click_coords = (x, y)
        elif current_mode == 'DEFINE_ZONE_START': first_corner = (x, y); print(f"-> Vertice 1 '{zone_names[current_zone_index]}' en {first_corner}"); current_mode = 'DEFINE_ZONE_END'
        elif current_mode == 'DEFINE_ZONE_END':
            if first_corner:
                x1=min(first_corner[0],x); y1=min(first_corner[1],y); x2=max(first_corner[0],x); y2=max(first_corner[1],y)
                zone_name = zone_names[current_zone_index]; defined_zones[zone_name] = (x1, y1, x2, y2); print(f"✅ Zona '{zone_name}' definida: ({x1},{y1}) a ({x2},{y2})")
                current_zone_index += 1; first_corner = None
                if current_zone_index < zones_to_define: current_mode = 'DEFINE_ZONE_START'
                else: print("✅ Todas las zonas definidas."); current_mode = 'PROCESSING';
            else: print("⚠️ Clic 2 sin Clic 1.")
if not args.non_interactive:
    # --- Fase 1: Seleccion del Docente ---
    print("ℹ️ Preparando seleccion..."); ret, frame_inicial_temp = cap.read()
    if ret:
        frame_inicial = frame_inicial_temp.copy(); results_primer_frame = proc.detectar_objetos(modelo_yolo, frame_inicial, conf=confianza_yolo, device=device_type)
        if results_primer_frame and hasattr(results_primer_frame[0], 'plot'): frame_display_inicial = results_primer_frame[0].plot()
        else: frame_display_inicial = frame_inicial
        cv2.namedWindow(nombre_ventana_unica, cv2.WINDOW_NORMAL); cv2.imshow(nombre_ventana_unica, frame_display_inicial); cv2.setMouseCallback(nombre_ventana_unica, on_mouse_click)
        print("➡️ Clic en docente. 'q' para salir.");
        while not tracking_activado:
            key = cv2.waitKey(20) & 0xFF;
            if key == ord("q"): cap.release(); cv2.destroyAllWindows(); sys.exit()
            if last_click_coords is not None and current_mode == 'SELECT_TEACHER':
                click_x, click_y = last_click_coords; last_click_coords = None; print(f"Clic en ({click_x}, {click_y})..."); found_teacher = False
                if results_primer_frame and hasattr(results_primer_frame[0], 'masks') and results_primer_frame[0].masks is not None:
                    for mask_tensor, box_tensor in zip(results_primer_frame[0].masks.data, results_primer_frame[0].boxes.data):
                        box_list = box_tensor.tolist();
                        if len(box_list) >= 6 and int(box_list[5]) == 0:
                            mask_np = mask_tensor.cpu().numpy(); mask_resized = cv2.resize(mask_np, (ancho, alto), interpolation=cv2.INTER_NEAREST)
                            if 0 <= click_y < alto and 0 <= click_x < ancho and mask_resized[click_y, click_x] > 0.5:
                                docente_mask_actual = mask_resized; tracking_activado = True; print("✅ Docente seleccionado."); current_mode = 'DEFINE_ZONE_START'; current_zone_index = 0; found_teacher = True; break
                if not found_teacher: print("    Clic no coincidio.");
            try:
                if cv2.getWindowProperty(nombre_ventana_unica, cv2.WND_PROP_VISIBLE) < 1: cap.release(); sys.exit()
            except cv2.error: cap.release(); sys.exit()
    else: print("❌ Error primer frame."); cap.release(); sys.exit()
    # --- Fase 2: Definicion Interactiva de Zonas ---
    print("\n--- Fase de Definicion de Zonas ---"); cv2.setMouseCallback(nombre_ventana_unica, on_mouse_click)
    while current_mode != 'PROCESSING':
        display_frame_zones = frame_inicial.copy(); proc.dibujar_zonas_definidas(display_frame_zones, defined_zones, color=(0, 255, 0), thickness=2)
        if first_corner: cv2.circle(display_frame_zones, first_corner, 5, (0, 0, 255), -1)
        prompt_text = "";
        if current_mode == 'DEFINE_ZONE_START': prompt_text = f"Define '{zone_names[current_zone_index]}': Clic Sup-Izq"
        elif current_mode == 'DEFINE_ZONE_END': prompt_text = f"Define '{zone_names[current_zone_index]}': Clic Inf-Der"
        cv2.putText(display_frame_zones, prompt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame_zones, "'q' para salir", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(nombre_ventana_unica, display_frame_zones); key = cv2.waitKey(30) & 0xFF;
        if key == ord("q"): print("Salida solicitada."); cap.release(); cv2.destroyAllWindows(); sys.exit()
        try:
            if cv2.getWindowProperty(nombre_ventana_unica, cv2.WND_PROP_VISIBLE) < 1: print(f"Ventana cerrada."); cap.release(); sys.exit()
        except cv2.error: print(f"Error ventana."); cap.release(); sys.exit()
else: # MODO NO INTERACTIVO
    print("ℹ️ Ejecutando en modo no interactivo.")
    ret, frame_inicial_temp = cap.read()
    if ret:
        frame_inicial = frame_inicial_temp.copy()
        print("    Intentando auto-seleccionar docente...")
        results_primer_frame = proc.detectar_objetos(modelo_yolo, frame_inicial, conf=confianza_yolo, device=device_type)
        initial_teacher_mask = None
        if results_primer_frame and hasattr(results_primer_frame[0], 'masks') and results_primer_frame[0].masks is not None and len(results_primer_frame[0].masks.data) > 0:
            print(f"    Detectadas {len(results_primer_frame[0].masks.data)} mascaras en primer frame.")
            first_person_found = False
            for mask_tensor, box_tensor in zip(results_primer_frame[0].masks.data, results_primer_frame[0].boxes.data):
                box_list = box_tensor.tolist();
                if len(box_list) >= 6 and int(box_list[5]) == 0:
                    mask_np = mask_tensor.cpu().numpy()
                    initial_teacher_mask = cv2.resize(mask_np, (ancho, alto), interpolation=cv2.INTER_NEAREST)
                    first_person_found = True
                    break
            if first_person_found:
                print("    ✅ Auto-seleccionado primer docente detectado.")
            else:
                print("    ⚠️ No se detectaron personas (clase 0) en el primer frame.")
        else:
            print("    ⚠️ No se encontraron mascaras en el primer frame.")
        if initial_teacher_mask is not None:
            docente_mask_actual = initial_teacher_mask
            tracking_activado = True
            print("    Tracking activado para docente auto-seleccionado.")
        else:
            print("    No se pudo seleccionar docente, tracking permanecera inactivo.")
            tracking_activado = False
    else: print("❌ Error leyendo primer frame en modo no interactivo."); cap.release(); sys.exit(1)
    print("    Usando Zonas Predefinidas (4 cuadrantes).")
    mid_x = ancho // 2; mid_y = alto // 2
    defined_zones = {"Arriba-Izq": (0, 0, mid_x, mid_y), "Arriba-Der": (mid_x, 0, ancho, mid_y), "Abajo-Izq": (0, mid_y, mid_x, alto), "Abajo-Der": (mid_x, mid_y, ancho, alto)}
    current_mode = 'PROCESSING'

# --- Inicializar contadores y mascaras de zona ---
print("DEBUG: Antes de inicializar contadores/mascaras de zona.")
zone_init_error = False
if defined_zones:
    print("    Inicializando contadores y mascaras de zona...")
    try: zone_frame_counts = {name: 0 for name in defined_zones.keys()}; zone_masks = {name: crear_mascara_zona(coords, alto, ancho) for name, coords in defined_zones.items()}; indefinido_frame_count = 0; print("    Contadores y mascaras listos.")
    except Exception as e_init: print(f"❌❌ ERROR INICIALIZACION ZONAS: {e_init}"); zone_frame_counts = {}; zone_masks = {}; indefinido_frame_count = 0; zone_init_error = True
else: print("⚠️ No se definieron zonas."); zone_frame_counts = {}; zone_masks = {}; indefinido_frame_count = 0
print("DEBUG: Despues de inicializar contadores/mascaras de zona.")

# --- Fase 3: Bucle Principal de Procesamiento ---
if not args.non_interactive:
    print("🖱️ Callback desactivado."); cv2.setMouseCallback(nombre_ventana_unica, lambda *args: None)
print(f"⏳ Iniciando procesamiento principal...")
start_time_processing_loop = time.time()
frames_processed_in_loop = 0 # Contador para progreso

while cap.isOpened():
    current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if current_frame_num > end_frame_num and end_frame_num != -1 : print(f"ℹ️ Alcanzado frame final ({end_frame_num})."); break
    
    ret, frame = cap.read()
    if not ret: print("ℹ️ Fin del video."); break
    
    frame_read_counter += 1
    
    process_this_frame = (frame_skip <= 0) or (frame_read_counter % (frame_skip + 1) == 1)
    
    if not process_this_frame:
        if not args.non_interactive and last_processed_overlay is not None:
            cv2.imshow(nombre_ventana_unica, last_processed_overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): print("Salida solicitada."); break
        if not args.non_interactive:
            try:
                if cv2.getWindowProperty(nombre_ventana_unica, cv2.WND_PROP_VISIBLE) < 1: print(f"Ventana cerrada."); break
            except cv2.error: print(f"Error ventana."); break
        continue

    frames_processed_in_loop += 1
    print(f"PROGRESO:{frames_processed_in_loop}/{total_frames_to_process:.0f}", flush=True)

    frames_procesados_post_seleccion += 1; start_time_frame = time.time()
    depth_map = proc.estimar_profundidad(modelo_profundidad, procesador_profundidad, frame, alto, ancho, device=device_type)
    results_loop = proc.detectar_objetos(modelo_yolo, frame, conf=confianza_yolo, device=device_type)
    docente_mask_actual, heatmap_posicion_acumulado, personas_detectadas, tupla_docente_frame = proc.procesar_detecciones_y_tracking(
        results_loop, tracking_activado, docente_mask_actual, heatmap_posicion_acumulado, ancho, alto, iou_tracking_threshold)
    
    total_personas_detectadas_acum += len(personas_detectadas)
    if tupla_docente_frame is not None:
        frames_con_docente_trackeado += 1
        if zone_masks and not zone_init_error:
            docente_mask_bool = (tupla_docente_frame[0] > 0.5); max_overlap_area = 0; best_zone_name = None
            for zone_name, zone_mask_bool in zone_masks.items():
                intersection_area = np.sum(np.logical_and(docente_mask_bool, zone_mask_bool))
                if intersection_area > max_overlap_area: max_overlap_area = intersection_area; best_zone_name = zone_name
            if best_zone_name is not None and max_overlap_area > 0 : zone_frame_counts[best_zone_name] += 1
            else: indefinido_frame_count += 1
        else: indefinido_frame_count +=1
        distancias_frame = proc.calcular_distancias_docente_otros(tupla_docente_frame, personas_detectadas)
        if distancias_frame:
            min_dist_frame = np.min(distancias_frame); lista_distancias_minimas.append(min_dist_frame)
            if min_dist_frame < threshold_cercania_px: frames_cercano_a_alguien += 1
            count_close_frame = sum(1 for d in distancias_frame if d < threshold_cercania_px); lista_personas_cercanas_count.append(count_close_frame)
    elif tracking_activado: indefinido_frame_count += 1
    avg_depth_teacher = None
    if docente_mask_actual is not None and depth_map is not None:
        idx_docente = docente_mask_actual > 0.5
        if np.any(idx_docente):
            depth_values_in_mask = depth_map[idx_docente]; valid_depths = depth_values_in_mask[np.isfinite(depth_values_in_mask)]
            if len(valid_depths) > 0: avg_depth_teacher = np.mean(valid_depths); lista_profundidades_docente.append(avg_depth_teacher)
            if tupla_docente_frame is not None and avg_depth_teacher is not None:
                box_doc_list = tupla_docente_frame[1].tolist(); x1d, y1d, x2d, y2d = box_doc_list[:4]; cx = int((x1d + x2d) / 2); cy = int((y1d + y2d) / 2)
                teacher_path.append((cx, cy, avg_depth_teacher))
    
    if not args.non_interactive:
        frame_display = frame.copy(); proc.dibujar_distancias_desde_docente(frame_display, tupla_docente_frame, personas_detectadas)
        proc.dibujar_zonas_definidas(frame_display, defined_zones, color=(0, 0, 255), thickness=2)
        overlay_final = proc.crear_overlay_con_heatmap(frame_display, heatmap_posicion_acumulado)
        last_processed_overlay = overlay_final.copy()
        cv2.imshow(nombre_ventana_unica, overlay_final)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): print("Salida solicitada."); break
        try:
            if cv2.getWindowProperty(nombre_ventana_unica, cv2.WND_PROP_VISIBLE) < 1: print(f"Ventana cerrada."); break
        except cv2.error: print(f"Error ventana."); break

# --- Limpieza, Cálculo de Métricas y Guardado Final ---
end_time_script = time.time(); print("Finalizando ejecucion.")
metricas = {}; metricas_zonales_str = []; metricas_audio_str = []; metricas_proximidad_str = []; metricas_movimiento_str = []
# --- Cálculo de Métricas ---
if start_time_processing_loop is not None and frames_procesados_post_seleccion > 0:
    total_processing_time = end_time_script - start_time_processing_loop; avg_fps = frames_procesados_post_seleccion / total_processing_time if total_processing_time > 0 else 0; porcentaje_tracking = (frames_con_docente_trackeado / frames_procesados_post_seleccion) * 100 if frames_procesados_post_seleccion > 0 else 0; avg_personas = total_personas_detectadas_acum / frames_procesados_post_seleccion if frames_procesados_post_seleccion > 0 else 0
    metricas['Tiempo Procesamiento Bucle (s)'] = f"{total_processing_time:.2f}"; metricas['Frames Procesados'] = frames_procesados_post_seleccion; metricas['FPS Promedio (en bucle)'] = f"{avg_fps:.2f}"; metricas['Frames Docente Trackeado'] = frames_con_docente_trackeado; metricas['Porcentaje Tracking Exitoso'] = f"{porcentaje_tracking:.1f}%"; metricas['Personas Promedio Detectadas'] = f"{avg_personas:.1f}"
    if lista_profundidades_docente: min_d = np.min(lista_profundidades_docente); max_d = np.max(lista_profundidades_docente); avg_d = np.mean(lista_profundidades_docente); metricas['Profundidad Relativa Docente (Min)'] = f"{min_d:.3f}"; metricas['Profundidad Relativa Docente (Max)'] = f"{max_d:.3f}"; metricas['Profundidad Relativa Docente (Avg)'] = f"{avg_d:.3f}"
    else: metricas['Profundidad Relativa Docente'] = "N/A"
    titulo_zonal = "\n--- Analisis Zonal (Basado en Max Overlap por Frame) ---"; print(titulo_zonal); metricas_zonales_str.append(titulo_zonal)
    if not defined_zones: linea_zonal = "- No se definieron zonas."; print(linea_zonal); metricas_zonales_str.append(linea_zonal)
    elif zone_init_error: linea_zonal = "- Error inicializando zonas."; print(linea_zonal); metricas_zonales_str.append(linea_zonal)
    elif not zone_frame_counts: linea_zonal = "- Error: Contadores zona vacios."; print(linea_zonal); metricas_zonales_str.append(linea_zonal)
    elif frames_con_docente_trackeado == 0: linea_zonal = "- No se trackeo al docente."; print(linea_zonal); metricas_zonales_str.append(linea_zonal)
    else:
        for zone_name, frame_count in zone_frame_counts.items(): porcentaje_zona = (frame_count / frames_con_docente_trackeado) * 100; linea_zonal = f"- Tiempo en Zona ({zone_name}): {porcentaje_zona:.1f}% ({frame_count} frames)"; metricas[f'Tiempo en Zona ({zone_name})'] = f"{porcentaje_zona:.1f}%"; print(linea_zonal); metricas_zonales_str.append(linea_zonal)
        porcentaje_indefinido = (indefinido_frame_count / frames_con_docente_trackeado) * 100; linea_indef = f"- Tiempo en Zona Indefinida: {porcentaje_indefinido:.1f}% ({indefinido_frame_count} frames)"; metricas['Tiempo en Zona Indefinida'] = f"{porcentaje_indefinido:.1f}%"; print(linea_indef); metricas_zonales_str.append(linea_indef)
    titulo_prox = "\n--- Analisis de Proximidad ---"; print(titulo_prox); metricas_proximidad_str.append(titulo_prox)
    if lista_distancias_minimas:
        avg_min_dist = np.mean(lista_distancias_minimas); min_min_dist = np.min(lista_distancias_minimas); porc_tiempo_cercano = (frames_cercano_a_alguien / frames_con_docente_trackeado) * 100 if frames_con_docente_trackeado > 0 else 0; avg_pers_cercanas = np.mean(lista_personas_cercanas_count) if lista_personas_cercanas_count else 0
        linea_prox1 = f"- Distancia Minima Promedio (a alguien): {avg_min_dist:.1f} px"; linea_prox2 = f"- % Tiempo Cerca (< {threshold_cercania_px} px de alguien): {porc_tiempo_cercano:.1f}%"; linea_prox3 = f"- No Promedio Personas Cercanas (< {threshold_cercania_px} px): {avg_pers_cercanas:.1f}"; metricas['DistMinPromedio_px'] = f"{avg_min_dist:.1f}"; metricas['%TiempoCerca'] = f"{porc_tiempo_cercano:.1f}%"; metricas['AvgPersCercanas'] = f"{avg_pers_cercanas:.1f}"
        print(linea_prox1); print(linea_prox2); print(linea_prox3); metricas_proximidad_str.extend([linea_prox1, linea_prox2, linea_prox3])
    else: linea_prox = "- No hay datos para analisis de proximidad."; print(linea_prox); metricas_proximidad_str.append(linea_prox)
    titulo_mov = "\n--- Analisis de Movimiento ---"; print(titulo_mov); metricas_movimiento_str.append(titulo_mov)
    if len(teacher_path) > 1 and frames_con_docente_trackeado > 0:
        frames_quieto = 0; distancia_total_px = 0.0
        for i in range(len(teacher_path) - 1): p1 = teacher_path[i]; p2 = teacher_path[i+1]; displacement = np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2])); distancia_total_px += displacement;
        if displacement < threshold_quietud_px: frames_quieto += 1
        porc_tiempo_quieto = (frames_quieto / frames_con_docente_trackeado) * 100; porc_tiempo_movimiento = 100.0 - porc_tiempo_quieto
        linea_mov1 = f"- Distancia Recorrida Total (Trayectoria): {distancia_total_px:.1f} px"; linea_mov2 = f"- % Tiempo Quieto (< {threshold_quietud_px} px/frame): {porc_tiempo_quieto:.1f}%"; linea_mov3 = f"- % Tiempo en Movimiento: {porc_tiempo_movimiento:.1f}%"; metricas['DistTotal_px'] = f"{distancia_total_px:.1f}"; metricas['%TiempoQuieto'] = f"{porc_tiempo_quieto:.1f}%"; metricas['%TiempoMovimiento'] = f"{porc_tiempo_movimiento:.1f}%"
        print(linea_mov1); print(linea_mov2); print(linea_mov3); metricas_movimiento_str.extend([linea_mov1, linea_mov2, linea_mov3])
    else: linea_mov = "- No hay datos para analisis de movimiento."; print(linea_mov); metricas_movimiento_str.append(linea_mov)
    titulo_audio = "\n--- Analisis de Audio (Transcripcion) ---"; print(titulo_audio); metricas_audio_str.append(titulo_audio)
    if transcript_available:
        teacher_talk_time_interval = 0.0; student_talk_time_interval = 0.0; other_talk_time_interval = 0.0; total_word_count_interval = 0; teacher_word_count_interval = 0; teacher_id = 'SPEAKER_00'
        analysis_start_time_sec = start_time_sec; analysis_end_time_sec = end_time_sec if end_time_sec > 0 else video_total_duration_sec
        if analysis_end_time_sec <= 0: analysis_end_time_sec = float('inf'); print("Advertencia: Duracion total video no determinada.")
        print(f"Analizando audio entre {analysis_start_time_sec:.2f}s y {analysis_end_time_sec:.2f}s")
        for index, row in transcript_df.iterrows():
            start_csv = row['start']; end_csv = row['end']; speaker = row['speaker']; segment_text = row['text']
            if pd.isna(start_csv) or pd.isna(end_csv) or end_csv <= start_csv: continue
            overlap_start = max(start_csv, analysis_start_time_sec); overlap_end = min(end_csv, analysis_end_time_sec); overlap_duration = max(0.0, overlap_end - overlap_start)
            if overlap_duration > 0:
                is_teacher = pd.notna(speaker) and speaker == teacher_id
                if is_teacher: teacher_talk_time_interval += overlap_duration
                elif pd.notna(speaker): student_talk_time_interval += overlap_duration
                else: other_talk_time_interval += overlap_duration
                if pd.notna(segment_text) and isinstance(segment_text, str): words_in_segment = segment_text.split(); current_segment_word_count = len(words_in_segment); total_word_count_interval += current_segment_word_count
                if is_teacher and pd.notna(segment_text) and isinstance(segment_text, str): teacher_word_count_interval += current_segment_word_count
        total_talk_time_in_interval = teacher_talk_time_interval + student_talk_time_interval + other_talk_time_interval
        if total_talk_time_in_interval > 0 or total_word_count_interval > 0 :
            linea_palabras_tot = f"- Palabras Totales (en intervalo): {total_word_count_interval}"; linea_palabras_doc = f"- Palabras Docente (en intervalo): {teacher_word_count_interval}"; metricas['Palabras_intervalo_Total'] = total_word_count_interval; metricas['Palabras_intervalo_Docente'] = teacher_word_count_interval; print(linea_palabras_tot); print(linea_palabras_doc); metricas_audio_str.extend([linea_palabras_tot, linea_palabras_doc])
            if total_talk_time_in_interval > 0:
                perc_ttt = (teacher_talk_time_interval / total_talk_time_in_interval) * 100; perc_stt_otros = (student_talk_time_interval + other_talk_time_interval) / total_talk_time_in_interval * 100
                linea1 = f"- Tiempo Habla Docente (TTT intervalo): {teacher_talk_time_interval:.1f} s"; linea2 = f"- Tiempo Habla Otros (STT+Otros intervalo): {student_talk_time_interval + other_talk_time_interval:.1f} s"; linea3 = f"- Porcentaje TTT (sobre habla intervalo): {perc_ttt:.1f}%"; linea4 = f"- Porcentaje STT+Otros (sobre habla intervalo): {perc_stt_otros:.1f}%"; metricas['TTT_intervalo (s)'] = f"{teacher_talk_time_interval:.1f}"; metricas['STT+Otros_intervalo (s)'] = f"{student_talk_time_interval + other_talk_time_interval:.1f}"; metricas['% TTT_intervalo'] = f"{perc_ttt:.1f}%"; metricas['% STT+Otros_intervalo'] = f"{perc_stt_otros:.1f}%"; print(linea1); print(linea2); print(linea3); print(linea4); metricas_audio_str.extend([linea1, linea2, linea3, linea4])
            else: linea_audio = "- No se encontro tiempo de habla valido."; print(linea_audio); metricas_audio_str.append(linea_audio)
        else: linea_audio = "- No se encontro habla ni palabras en intervalo analizado."; print(linea_audio); metricas_audio_str.append(linea_audio)
    else: linea_audio = "- Transcripcion no disponible/invalida."; print(linea_audio); metricas_audio_str.append(linea_audio)
else: metricas['Resultado'] = "No se procesaron frames."
print("\n" + "="*15 + " METRICAS GENERALES " + "="*15)
exclude_prefixes = ('Tiempo en Zona', 'TTT', 'STT', '% TTT', '% STT', 'Palabras', 'DistMin', '%TiempoCerca', 'AvgPersCercanas', 'DistTotal', '%TiempoQuieto', '%TiempoMovimiento')
for key, value in metricas.items():
    if not any(key.startswith(prefix) for prefix in exclude_prefixes): print(f"- {key}: {value}")
print("="*53)
try:
    ruta_guardar_metricas = os.path.join(run_output_dir, "metricas.txt")
    with open(ruta_guardar_metricas, 'w', encoding='utf-8') as f:
        f.write(f"Metricas para video: {nombre_video_base} (Intervalo: {start_time_sec:.1f}s - {end_time_sec:.1f}s)\n"); f.write("="*30 + "\n")
        for key, value in metricas.items():
            if not any(key.startswith(prefix) for prefix in exclude_prefixes): f.write(f"- {key}: {value}\n")
        for linea_zonal in metricas_zonales_str: f.write(linea_zonal + "\n")
        for linea_prox in metricas_proximidad_str: f.write(linea_prox + "\n")
        for linea_mov in metricas_movimiento_str: f.write(linea_mov + "\n")
        for linea_audio in metricas_audio_str: f.write(linea_audio + "\n")
    print(f"✅ Metricas guardadas en: {ruta_guardar_metricas}")
except Exception as e: print(f"❌ Error al guardar metricas: {e}")
if frame_inicial is not None:
    zone_draw_color_save = (0, 0, 0); zone_draw_thickness_save = 2
    imagen_heatmap_pos = proc.crear_imagen_heatmap(heatmap_posicion_acumulado, background_img=frame_inicial, zones_to_draw=defined_zones, zone_color=zone_draw_color_save, zone_thickness=zone_draw_thickness_save)
    if imagen_heatmap_pos is not None:
        max_heatmap_val = np.max(heatmap_posicion_acumulado);
        if max_heatmap_val > 0: imagen_heatmap_pos_con_barra = proc.agregar_colorbar(imagen_heatmap_pos, cmap_name_mpl='jet', vmin=0, vmax=max_heatmap_val, etiqueta='Tiempo Acumulado (Azul=Min, Rojo=Max)')
        else: imagen_heatmap_pos_con_barra = imagen_heatmap_pos
        ruta_guardar_pos = os.path.join(run_output_dir, "heatmap_posicion.png");
        try: cv2.imwrite(ruta_guardar_pos, imagen_heatmap_pos_con_barra); print(f"✅ Heatmap de posicion guardado en: {ruta_guardar_pos}")
        except Exception as e: print(f"❌ Error al guardar heatmap de posicion: {e}")
    else: print("⚠️ No se genero imagen de heatmap de posicion.")
else: print("⚠️ No se guardo heatmap de posicion con fondo.")
if len(teacher_path) > 1:
    ruta_guardar_traj = os.path.join(run_output_dir, "trayectoria.png")
    trajectory_image, min_d, max_d = proc.dibujar_trayectoria_coloreada(teacher_path, ancho, alto)
    if trajectory_image is not None:
        proc.dibujar_zonas_definidas(trajectory_image, defined_zones, color=(150, 150, 150), thickness=1)
        if min_d is not None and max_d is not None: imagen_traj_con_barra = proc.agregar_colorbar(trajectory_image, cmap_name_mpl='jet', vmin=min_d, vmax=max_d, etiqueta='Profundidad Relativa (Azul=Cerca, Rojo=Lejos)')
        else: imagen_traj_con_barra = trajectory_image
        try: cv2.imwrite(ruta_guardar_traj, imagen_traj_con_barra); print(f"✅ Imagen de trayectoria guardada en: {ruta_guardar_traj}")
        except Exception as e: print(f"❌ Error al guardar imagen de trayectoria: {e}")
    else: print("⚠️ No se genero imagen de trayectoria.")
else: print("ℹ️ No hay suficientes datos de trayectoria.")
cap.release()
if not args.non_interactive: cv2.destroyAllWindows()
print("✅ Recursos liberados. Fin del programa.")