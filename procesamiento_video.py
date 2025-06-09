# procesamiento_video.py
# ... (imports iguales que antes: cv2, np, os, torch, F, plt, mpl, BytesIO, PILImage, YOLO, transformers) ...
import cv2
import numpy as np
import os
from itertools import combinations
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from io import BytesIO
from PIL import Image as PILImage # Usar Pillow

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError:
    print("⚠️ Advertencia: No se encontraron 'transformers'. Instala con: pip install transformers")
    AutoImageProcessor, AutoModelForDepthEstimation = None, None

# --- Función auxiliar para crear máscaras de zona ---
def crear_mascara_zona(coords, h, w):
    # ... (Sin cambios - versión corregida) ...
    mask_uint8 = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, coords)
    x1_safe, y1_safe = max(0, x1), max(0, y1); x2_safe, y2_safe = min(w, x2), min(h, y2)
    if x1_safe < x2_safe and y1_safe < y2_safe: cv2.rectangle(mask_uint8, (x1_safe, y1_safe), (x2_safe, y2_safe), 255, -1)
    mask_bool = (mask_uint8 > 0); return mask_bool

# --- Funciones de Lógica Central ---
def calcular_iou(mask1, mask2):
    # ... (Sin cambios) ...
    mask1_bool = mask1 > 0; mask2_bool = mask2 > 0
    inter = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    return inter / union if union > 0 else 0

def calcular_distancia(box1, box2):
    # ... (Sin cambios) ...
    x1a, y1a, x2a, y2a = box1.tolist()[:4]; x1b, y1b, x2b, y2b = box2.tolist()[:4]
    centro_a = ((x1a + x2a) / 2, (y1a + y2a) / 2); centro_b = ((x1b + x2b) / 2, (y1b + y2b) / 2)
    distancia = np.linalg.norm(np.array(centro_a) - np.array(centro_b))
    return distancia, centro_a, centro_b

# --- NUEVA Función para Calcular Distancias Docente-Otros ---
def calcular_distancias_docente_otros(tupla_docente, lista_otras_personas):
    """
    Calcula una lista de distancias (en píxeles) desde el docente
    hacia todas las otras personas detectadas.
    """
    distancias = []
    if tupla_docente is None or not lista_otras_personas or len(lista_otras_personas) <= 1:
        return distancias # No hay docente o no hay otros

    mask_docente, box_docente = tupla_docente
    box_docente_list = box_docente.tolist() # Para comparación

    for mask_otra, box_otra in lista_otras_personas:
        box_otra_list = box_otra.tolist()
        # Evitar calcular distancia del docente a sí mismo
        if box_docente_list == box_otra_list:
            continue

        # Calcular distancia y añadirla a la lista
        distancia, _, _ = calcular_distancia(box_docente, box_otra)
        distancias.append(distancia)

    return distancias
# --- Fin Nueva Función ---


# --- Funciones de Procesamiento YOLO ---
# (cargar_modelo_yolo, detectar_objetos sin cambios)
# ... (igual que antes) ...
def cargar_modelo_yolo(ruta_modelo="yolo11l-seg.pt"):
    print(f"🔄 Cargando modelo YOLO: {ruta_modelo}...")
    try: model = YOLO(ruta_modelo); print("✅ Modelo YOLO cargado."); return model
    except Exception as e: print(f"❌ Error cargando YOLO: {e}"); return None

def detectar_objetos(modelo, frame, conf=0.25, device='cpu'):
    if modelo is None: return None
    try: return modelo(frame, conf=conf, device=device)
    except Exception as e: print(f"❌ Error detección YOLO: {e}"); return None


# --- Funciones de Tracking y Heatmap ---
# (procesar_detecciones_y_tracking sin cambios)
# ... (igual que antes) ...
def procesar_detecciones_y_tracking(results_loop, tracking_activado, docente_mask_in, heatmap_posicion_in, ancho, alto, iou_threshold=0.3):
    personas = []; docente_mask_out = docente_mask_in; heatmap_posicion_out = heatmap_posicion_in
    mascara_mejor_candidato = None; box_mejor_candidato = None; mejor_iou = 0.0
    tupla_docente_identificado = None
    if results_loop and hasattr(results_loop[0], 'masks') and results_loop[0].masks is not None:
        for mask, box in zip(results_loop[0].masks.data, results_loop[0].boxes.data):
            box_list = box.tolist()
            if len(box_list) >= 6:
                cls = int(box_list[5])
                if cls == 0:
                    mask_resized = cv2.resize(mask.cpu().numpy(), (ancho, alto), interpolation=cv2.INTER_NEAREST)
                    personas.append((mask_resized, box))
                    if tracking_activado and docente_mask_in is not None:
                        iou = calcular_iou(docente_mask_in, mask_resized)
                        if iou > mejor_iou: mejor_iou = iou; mascara_mejor_candidato = mask_resized; box_mejor_candidato = box
    if tracking_activado and mascara_mejor_candidato is not None and mejor_iou > iou_threshold:
        docente_mask_out = mascara_mejor_candidato
        heatmap_posicion_out = heatmap_posicion_in + (docente_mask_out > 0.5).astype(np.float32)
        tupla_docente_identificado = (mascara_mejor_candidato, box_mejor_candidato)
    return docente_mask_out, heatmap_posicion_out, personas, tupla_docente_identificado

# --- Funciones para Depth Anything ---
# (cargar_modelo_profundidad, estimar_profundidad sin cambios)
# ... (igual que antes) ...
def cargar_modelo_profundidad(model_name="LiheYoung/depth-anything-small-hf", device='cpu'):
    if AutoImageProcessor is None or AutoModelForDepthEstimation is None: print("❌ Error: Librería 'transformers' no encontrada."); return None, None
    print(f"🔄 Cargando modelo de profundidad: {model_name}...")
    try:
        processor = AutoImageProcessor.from_pretrained(model_name); model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device); model.eval()
        print("✅ Modelo de profundidad cargado."); return model, processor
    except Exception as e: print(f"❌ Error al cargar modelo de profundidad: {e}"); return None, None

def estimar_profundidad(modelo_depth, processor_depth, frame, target_height, target_width, device='cpu'):
    if modelo_depth is None or processor_depth is None: return None
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); inputs = processor_depth(images=frame_rgb, return_tensors="pt").to(device)
        with torch.no_grad(): outputs = modelo_depth(**inputs); predicted_depth = outputs.predicted_depth
        if predicted_depth.ndim == 3: predicted_depth = predicted_depth.unsqueeze(0)
        elif predicted_depth.ndim == 2: predicted_depth = predicted_depth.unsqueeze(0).unsqueeze(0)
        prediction = F.interpolate(predicted_depth, size=(target_height, target_width), mode="bicubic", align_corners=False)
        depth_map = prediction.squeeze().cpu().numpy(); return depth_map
    except Exception as e: print(f"❌ Error durante estimación de profundidad: {e}"); return None

# --- Funciones de Visualización ---
# (dibujar_distancias_desde_docente, crear_overlay_con_heatmap, dibujar_trayectoria_coloreada, dibujar_zonas_definidas, agregar_colorbar, crear_imagen_heatmap sin cambios)
# ... (igual que antes) ...
def dibujar_distancias_desde_docente(frame, tupla_docente, lista_otras_personas):
     if tupla_docente is None or not lista_otras_personas or len(lista_otras_personas) <= 1: return
     mask_docente, box_docente = tupla_docente; box_docente_list = box_docente.tolist()
     for mask_otra, box_otra in lista_otras_personas:
         box_otra_list = box_otra.tolist()
         if box_docente_list == box_otra_list: continue
         distancia, centro_docente, centro_otra = calcular_distancia(box_docente, box_otra); texto = f"{distancia:.0f} px"
         cv2.line(frame, tuple(map(int, centro_docente)), tuple(map(int, centro_otra)), (0, 255, 255), 1)
         mid = (int((centro_docente[0] + centro_otra[0]) / 2), int((centro_docente[1] + centro_otra[1]) / 2))
         cv2.putText(frame, texto, (mid[0], mid[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

def crear_overlay_con_heatmap(frame_con_dibujos, heatmap_posicion):
    if np.max(heatmap_posicion) > 0:
        try:
            heatmap_norm = cv2.normalize(heatmap_posicion, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            if len(frame_con_dibujos.shape) == 2: frame_con_dibujos = cv2.cvtColor(frame_con_dibujos, cv2.COLOR_GRAY2BGR)
            if len(heatmap_color.shape) == 2: heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_GRAY2BGR)
            if frame_con_dibujos.shape == heatmap_color.shape: return cv2.addWeighted(frame_con_dibujos, 0.7, heatmap_color, 0.3, 0)
            else: print(f"⚠️ Shape mismatch: Frame {frame_con_dibujos.shape}, Heatmap {heatmap_color.shape}"); return frame_con_dibujos
        except Exception as e: print(f"❌ Error al crear overlay de heatmap: {e}"); return frame_con_dibujos
    else: return frame_con_dibujos

def dibujar_trayectoria_coloreada(path_data, img_width, img_height, cmap_name=cv2.COLORMAP_JET, thickness=2, background_color=(0, 0, 0)):
    background = np.full((img_height, img_width, 3), background_color, dtype=np.uint8); min_depth_traj, max_depth_traj = None, None
    if not path_data or len(path_data) < 2: print("ℹ️ No hay suficientes datos de trayectoria."); return background, min_depth_traj, max_depth_traj
    depths = np.array([item[2] for item in path_data if item[2] is not None and np.isfinite(item[2])])
    if len(depths) < 1: print("ℹ️ No hay datos de profundidad válidos para colorear trayectoria."); return background, min_depth_traj, max_depth_traj
    min_depth_traj = np.min(depths); max_depth_traj = np.max(depths); depth_range = max_depth_traj - min_depth_traj
    colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cmap_name)
    for i in range(len(path_data) - 1):
        x1, y1, d1 = path_data[i]; x2, y2, d2 = path_data[i+1]; current_depth = d1
        if current_depth is None or not np.isfinite(current_depth): color_tuple = (128, 128, 128)
        else:
            if depth_range < 1e-6: normalized_depth = 0.5
            else: normalized_depth = (current_depth - min_depth_traj) / depth_range
            normalized_depth = np.clip(normalized_depth, 0.0, 1.0); color_index = max(0, min(255, int(normalized_depth * 255)))
            b, g, r = colormap[color_index][0]; color_tuple = (int(b), int(g), int(r))
        p1 = (int(x1), int(y1)); p2 = (int(x2), int(y2)); cv2.line(background, p1, p2, color_tuple, thickness)
    # print(f"✅ Imagen de trayectoria generada con {len(path_data)} puntos.") # Quitamos este print
    return background, min_depth_traj, max_depth_traj

def dibujar_zonas_definidas(imagen, zonas, color=(255, 255, 255), thickness=1, font_scale=0.4):
    if zonas is None or not isinstance(zonas, dict) or not zonas: return
    for nombre, (x1, y1, x2, y2) in zonas.items():
        pt1 = (int(x1), int(y1)); pt2 = (int(x2), int(y2))
        try: cv2.rectangle(imagen, pt1, pt2, color, thickness); cv2.putText(imagen, nombre, (pt1[0] + 5, pt1[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        except Exception as e: print(f"      ERROR al dibujar zona '{nombre}': {e}")

def agregar_colorbar(imagen_np, cmap_name_mpl, vmin, vmax, etiqueta, bar_width_ratio=0.05, padding_ratio=0.02):
    try:
        img_h, img_w = imagen_np.shape[:2]; colorbar_width_px = int(img_w * bar_width_ratio); padding_px = int(img_w * padding_ratio); total_width = img_w + padding_px + colorbar_width_px
        dpi = 100; fig_height_in = img_h / dpi; fig_width_in = colorbar_width_px / dpi; fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax); cmap = plt.get_cmap(cmap_name_mpl)
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical'); cb.set_label(etiqueta, fontsize=8); cb.ax.tick_params(labelsize=7)
        buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0); buf.seek(0)
        colorbar_img_pil = PILImage.open(buf); plt.close(fig)
        colorbar_img_np = cv2.cvtColor(np.array(colorbar_img_pil), cv2.COLOR_RGBA2BGR)
        colorbar_resized = cv2.resize(colorbar_img_np, (colorbar_width_px, img_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((img_h, total_width, 3), 255, dtype=np.uint8); canvas[0:img_h, 0:img_w] = imagen_np; start_col = img_w + padding_px
        canvas[0:img_h, start_col : start_col + colorbar_width_px] = colorbar_resized; return canvas
    except Exception as e: print(f"❌ Error al agregar colorbar: {e}"); return imagen_np

# --- Funciones de Archivo ---
# Reemplaza la función original 'crear_imagen_heatmap' con esta versión mejorada

def crear_imagen_heatmap(heatmap_data, min_val=0, background_img=None, alpha=0.6, beta=0.4, zones_to_draw=None, zone_color=(255, 255, 255), zone_thickness=1):
    """
    Crea una imagen de mapa de calor con una mezcla selectiva para evitar que
    las zonas de baja actividad se vuelvan transparentes.
    """
    imagen_resultante = None
    # Asegurarse de que hay datos en el heatmap para procesar
    if np.max(heatmap_data) > min_val:
        try:
            # 1. Normalizar los datos y aplicar el colormap (igual que antes)
            norm_heatmap = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_color = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)

            # Usar la imagen de fondo si está disponible, sino, un fondo negro
            if background_img is not None and background_img.shape[:2] == heatmap_color.shape[:2]:
                # Asegurarse que el fondo sea un BGR de 3 canales
                background_processed = background_img.astype(np.uint8)
                if len(background_processed.shape) == 2:
                    background_processed = cv2.cvtColor(background_processed, cv2.COLOR_GRAY2BGR)
                imagen_resultante = background_processed.copy()
            else:
                if background_img is not None:
                    print("⚠️ Imagen de fondo no válida o con dimensiones incorrectas. Usando fondo negro.")
                imagen_resultante = np.zeros_like(heatmap_color)

            # --- LA LÓGICA CLAVE MEJORADA ---
            # 2. Crear una máscara donde el heatmap tiene datos (el docente estuvo presente)
            # La máscara será 1 donde heatmap_data > min_val, y 0 donde no.
            mask = (heatmap_data > min_val).astype(np.uint8)

            # 3. Aplicar el color del heatmap solo en la región de la máscara
            # Para cada pixel, si la máscara es > 0, se aplica la mezcla. Si no, se queda el pixel del fondo.
            
            # Extraer los píxeles de la región de interés (ROI) del fondo y del heatmap
            roi_background = imagen_resultante[mask > 0]
            roi_heatmap_color = heatmap_color[mask > 0]

            # Mezclar solo esos píxeles
            if roi_background.size > 0:
                blended_roi = cv2.addWeighted(roi_background, alpha, roi_heatmap_color, beta, 0.0)
                # Poner los píxeles mezclados de vuelta en la imagen final
                imagen_resultante[mask > 0] = blended_roi
            # --- FIN DE LA LÓGICA MEJORADA ---

            # Dibujar las zonas encima del resultado final, si se proporcionan
            if zones_to_draw is not None and imagen_resultante is not None:
                dibujar_zonas_definidas(imagen_resultante, zones_to_draw, color=zone_color, thickness=zone_thickness)

        except Exception as e:
            print(f"❌ Error al crear imagen de mapa de calor: {e}")
            imagen_resultante = None
    else:
        print(f"ℹ️ Heatmap vacío, no se genera imagen.")
        # Si no hay datos, pero se necesita un fondo, devolverlo
        if background_img is not None:
            imagen_resultante = background_img.copy()
            if zones_to_draw is not None:
                 dibujar_zonas_definidas(imagen_resultante, zones_to_draw, color=zone_color, thickness=zone_thickness)
    
    return imagen_resultante