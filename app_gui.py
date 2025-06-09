# app_gui.py (MODIFICADO - Ventana Expandible)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import configparser
import os
import cv2
import subprocess
import sys
import glob
import math
import re
import datetime

# Imports para PDF (con comprobación)
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportlabImage, PageBreak, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from PIL import Image as PILImage
    reportlab_available = True
except ImportError:
    print("⚠️ ADVERTENCIA: No se encontró 'reportlab' o 'Pillow'. La generación y apertura de PDF no funcionará.")
    print("      Instala con: pip install reportlab Pillow")
    reportlab_available = False

# --- Constantes para Rutas por Defecto ---
DEFAULT_VIDEO_DIR = r"C:\Users\Angel\Documents\Tesis\Seminario 2\Videostesis"
DEFAULT_TRANSCRIPT_DIR = r"C:\Users\Angel\Documents\Tesis\Seminario 2\Videostesis"
OUTPUT_BASE_DIR = r"C:\Users\Angel\Documents\Tesis\Seminario 2\Mapacalor"
PDF_FILENAME = "reporte.pdf"

# --- Variables Globales ---
selected_video = None; video_duration = 0.0; selected_transcript = None
running_process = None;

# --- Funciones de la GUI ---
def seleccionar_video():
    global selected_video, video_duration
    if running_process is not None and running_process.poll() is None: messagebox.showwarning("Proceso Activo", "Espera..."); return
    initial_dir = DEFAULT_VIDEO_DIR
    if selected_video and os.path.isdir(os.path.dirname(selected_video)): initial_dir = os.path.dirname(selected_video)
    elif not os.path.isdir(DEFAULT_VIDEO_DIR): initial_dir = os.path.expanduser("~")
    video_path = filedialog.askopenfilename( initialdir=initial_dir, title="Selecciona Video MP4", filetypes=[("Archivos MP4", "*.mp4"), ("Todos", "*.*")])
    if video_path:
        if not video_path.lower().endswith(".mp4"): messagebox.showwarning("Formato no preferido", "Archivo no MP4 seleccionado.")
        video_name = os.path.basename(video_path); video_label.config(text=f"🎥 {video_name}"); selected_video = video_path
        pdf_button.config(state=tk.DISABLED); pdf_open_button.config(state=tk.DISABLED)
        try:
            cap = cv2.VideoCapture(video_path);
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS); frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps > 0 and frame_count > 0:
                    video_duration = frame_count / fps; print(f"Video: {video_name}, Duración: {video_duration:.2f}s")
                    slider_start.config(to=video_duration); slider_end.config(to=video_duration); slider_start.set(0); slider_end.set(video_duration)
                    update_slider_labels(); process_button.config(state=tk.NORMAL); status_label.config(text="Listo para procesar.")
                    actualizar_historial(selected_video)
                    if history_tree.get_children() and reportlab_available:
                        pdf_button.config(state=tk.NORMAL)
                        pdf_open_button.config(state=tk.NORMAL)
                else: messagebox.showerror("Error", "No se pudo leer FPS/Frames."); process_button.config(state=tk.DISABLED); status_label.config(text="Error video.")
                cap.release()
            else: messagebox.showerror("Error", "No se pudo abrir video."); process_button.config(state=tk.DISABLED); status_label.config(text="Error video.")
        except Exception as e: messagebox.showerror("Error", f"Error procesando video: {e}"); process_button.config(state=tk.DISABLED); status_label.config(text="Error video.")

def seleccionar_transcripcion():
    global selected_transcript
    if running_process is not None and running_process.poll() is None: messagebox.showwarning("Proceso Activo", "Espera..."); return
    initial_dir = DEFAULT_TRANSCRIPT_DIR;
    if selected_transcript and os.path.isdir(os.path.dirname(selected_transcript)): initial_dir = os.path.dirname(selected_transcript)
    elif selected_video and os.path.isdir(os.path.dirname(selected_video)): initial_dir = os.path.dirname(selected_video)
    elif not os.path.isdir(DEFAULT_TRANSCRIPT_DIR): initial_dir = os.path.expanduser("~")
    transcript_path = filedialog.askopenfilename(initialdir=initial_dir, title="Selecciona CSV (Opcional)", filetypes=[("Archivos CSV", "*.csv"), ("Todos", "*.*")])
    if transcript_path: transcript_name = os.path.basename(transcript_path); transcript_label.config(text=f"📜 {transcript_name}"); selected_transcript = transcript_path; print(f"Transcripción: {transcript_name}")
    else: transcript_label.config(text="Ninguna transcripción seleccionada."); selected_transcript = None

def update_slider_labels(event=None):
    try:
        start_time_sec = slider_start.get(); end_time_sec = slider_end.get()
        start_minutes = int(start_time_sec // 60); start_seconds = int(start_time_sec % 60); start_display_str = f"{start_minutes:02d}:{start_seconds:02d}"
        end_minutes = int(end_time_sec // 60); end_seconds = int(end_time_sec % 60); end_display_str = f"{end_minutes:02d}:{end_seconds:02d}"
        start_label_val.config(text=start_display_str); end_label_val.config(text=end_display_str)
    except tk.TclError: pass

validating_interval = False
def validar_intervalo(event=None):
    global validating_interval
    if validating_interval: return
    validating_interval = True
    try:
        start_time = slider_start.get(); end_time = slider_end.get()
        if start_time >= (end_time - 0.1):
            new_start = max(0.0, end_time - 0.5);
            if new_start < slider_start['from']: new_start = slider_start['from']
            if abs(slider_start.get() - new_start) > 1e-6: slider_start.set(new_start)
        update_slider_labels()
    except tk.TclError: pass
    finally: validating_interval = False

def update_frame_skip_label(event=None):
    try:
        skip_val = int(frame_skip_var.get())
        frame_skip_val_label.config(text=f"{skip_val}")
    except tk.TclError:
        pass

def habilitar_controles():
    select_button.config(state=tk.NORMAL); transcript_button.config(state=tk.NORMAL)
    if selected_video: process_button.config(state=tk.NORMAL)
    status_label.config(text="Listo para procesar.")
    if history_tree.get_children() and reportlab_available:
        pdf_button.config(state=tk.NORMAL)
        pdf_open_button.config(state=tk.NORMAL)
    else:
        pdf_button.config(state=tk.DISABLED)
        pdf_open_button.config(state=tk.DISABLED)

def check_process_status():
    global running_process
    if running_process is None: habilitar_controles(); return
    if running_process.poll() is not None:
        print("--- Proceso main_opencv.py Terminado ---"); print(f"Código de salida: {running_process.poll()}")
        running_process = None;
        actualizar_historial(selected_video)
        habilitar_controles()
        messagebox.showinfo("Proceso Terminado", "Análisis finalizado.\nHistorial actualizado. Ya puedes generar o abrir el reporte en PDF.")
    else: root.after(1000, check_process_status)

def procesar_video_seleccionado():
    global running_process
    if not selected_video: messagebox.showerror("Error", "Selecciona un video."); return
    start_time = slider_start.get(); end_time = slider_end.get()
    if start_time >= end_time: messagebox.showerror("Error", "Tiempo inicio >= Tiempo fin."); return
    select_button.config(state=tk.DISABLED); transcript_button.config(state=tk.DISABLED); process_button.config(state=tk.DISABLED)
    pdf_button.config(state=tk.DISABLED); pdf_open_button.config(state=tk.DISABLED)
    status_label.config(text="⏳ Iniciando procesamiento..."); root.update_idletasks()
    try:
        python_exe = sys.executable
        main_script_path = "main_opencv.py"
        command = [
            python_exe, main_script_path,
            '--video', selected_video,
            '--start', str(start_time),
            '--end', str(end_time)
        ]
        if selected_transcript and os.path.exists(selected_transcript):
            command.extend(['--transcript', selected_transcript])
            print(f"Usando transcripción: {selected_transcript}")
        else:
            print("No se usará archivo de transcripción.")
        try:
            frame_skip_value = int(frame_skip_var.get())
            if frame_skip_value > 0:
                command.extend(['--frame-skip', str(frame_skip_value)])
                print(f"Usando Frame Skip: {frame_skip_value}")
        except (ValueError, tk.TclError):
            print("Valor de Frame Skip inválido, se usará 0 por defecto.")
        print("Ejecutando comando:", " ".join(command)); running_process = subprocess.Popen(command)
        status_label.config(text="⏳ Procesando en ventana separada..."); root.after(1000, check_process_status)
    except Exception as e: messagebox.showerror("Error al iniciar", f"No se pudo lanzar el proceso:\n{e}"); habilitar_controles(); running_process = None; return

def actualizar_historial(video_path):
    for i in history_tree.get_children(): history_tree.delete(i)
    if not video_path: history_tree.insert('', tk.END, values=("(Selecciona un video)", "", "")); return
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        if not os.path.isdir(OUTPUT_BASE_DIR): history_tree.insert('', tk.END, values=("Carpeta base no encontrada.", "", "")); return
        search_pattern = os.path.join(OUTPUT_BASE_DIR, f"{base_name}_Procesamiento_*")
        run_folders = sorted([d for d in glob.glob(search_pattern) if os.path.isdir(d)], key=os.path.getmtime, reverse=True)
        if run_folders:
            count = 0
            for folder_path in run_folders:
                folder_name = os.path.basename(folder_path)
                match = re.search(r"Procesamiento_(\d+) \((\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\)", folder_name)
                if match:
                    num_run, date_str_ymd, time_str_hms = match.groups()
                    time_str_hms = time_str_hms.replace('-', ':')
                    try: date_str_dmy = datetime.datetime.strptime(date_str_ymd, '%Y-%m-%d').strftime('%d-%m-%Y')
                    except ValueError: date_str_dmy = date_str_ymd
                    run_display = f"Procesamiento {num_run}"
                    tag = 'evenrow' if count % 2 == 0 else 'oddrow'
                    history_tree.insert('', tk.END, iid=folder_path, values=(run_display, date_str_dmy, time_str_hms), tags=(tag,))
                    count += 1
                else: history_tree.insert('', tk.END, iid=folder_path, values=(folder_name, "N/A", "N/A"), tags=('oddrow',))
        else: history_tree.insert('', tk.END, values=("No hay historial para este video.", "", ""))
    except Exception as e: print(f"Error al buscar historial: {e}"); history_tree.insert('', tk.END, values=("Error al buscar historial.", "", ""))

def abrir_carpeta_historial(event):
    selection = history_tree.selection()
    if not selection: return
    folder_path = selection[0]
    if folder_path and os.path.isdir(folder_path): print(f"Abriendo carpeta: {folder_path}"); abrir_archivo(folder_path)
    else: messagebox.showwarning("Ruta Inválida", "La ruta seleccionada no es una carpeta válida.")

def abrir_archivo(path):
    if not path or not os.path.exists(path): messagebox.showerror("Error", f"La ruta no existe:\n{path}"); return
    print(f"Intentando abrir: {path}")
    try:
        if sys.platform == "win32": os.startfile(path)
        elif sys.platform == "darwin": subprocess.Popen(["open", path])
        else: subprocess.Popen(["xdg-open", path])
    except Exception as e: messagebox.showerror("Error", f"No se pudo abrir la ruta:\n{e}")

def get_selected_run_dir():
    selection = history_tree.selection()
    if not selection: messagebox.showinfo("Información", "Primero selecciona una ejecución del historial."); return None
    if len(selection) > 1: messagebox.showwarning("Advertencia", "Por favor, selecciona solo UNA ejecución a la vez."); return None
    return selection[0]

def abrir_pdf_reporte():
    selected_run_dir = get_selected_run_dir()
    if not selected_run_dir: return
    pdf_path = os.path.join(selected_run_dir, PDF_FILENAME)
    if os.path.exists(pdf_path): abrir_archivo(pdf_path)
    else: messagebox.showinfo("Archivo no encontrado", f"'{PDF_FILENAME}' no existe en esta ejecución.\nGenera el reporte primero.")

def generar_pdf():
    if not reportlab_available: messagebox.showerror("Error", "Librerías PDF no instaladas."); return
    if not selected_video: messagebox.showerror("Error", "No hay video base seleccionado."); return
    selected_run_dir = get_selected_run_dir()
    if not selected_run_dir: return
    print(f"Generando PDF para: {selected_run_dir}")
    ruta_metricas = os.path.join(selected_run_dir, "metricas.txt"); ruta_pos = os.path.join(selected_run_dir, "heatmap_posicion.png"); ruta_traj = os.path.join(selected_run_dir, "trayectoria.png")
    if not all(os.path.exists(p) for p in [ruta_metricas, ruta_pos, ruta_traj]): messagebox.showerror("Error", "Faltan archivos en:\n" + selected_run_dir); return
    pdf_save_path = os.path.join(selected_run_dir, PDF_FILENAME)
    try:
        print(f"Generando PDF en: {pdf_save_path}"); doc = SimpleDocTemplate(pdf_save_path, pagesize=letter); styles = getSampleStyleSheet(); story = []
        video_name_short = os.path.splitext(os.path.basename(selected_video))[0]; run_time_str = os.path.basename(selected_run_dir).split('_', 1)[-1]; title = f"Reporte Análisis: {video_name_short}"; subtitle = f"Ejecución: {run_time_str}"; story.append(Paragraph(title, styles['h1'])); story.append(Paragraph(subtitle, styles['h2'])); story.append(Spacer(1, 0.2*inch))
        data_rendimiento = [['Métrica', 'Valor']]; data_espacial = [['Métrica', 'Valor']]; data_audio = [['Métrica', 'Valor']]; current_section = 'Rendimiento'
        try:
            with open(ruta_metricas, 'r', encoding='utf-8') as f: lines = f.readlines()
            for line in lines:
                line = line.strip();
                if not line or line.startswith("===") or line.startswith("Metricas para video"): continue
                if "--- Análisis Zonal" in line or "--- Análisis de Proximidad" in line or "--- Análisis de Movimiento" in line : current_section = 'Espacial'; continue
                elif "--- Análisis de Audio" in line: current_section = 'Audio'; continue
                elif line.startswith("---"): current_section = 'Rendimiento'; continue
                if line.startswith("- "):
                    parts = line[2:].split(':', 1)
                    if len(parts) == 2:
                        key, value = parts[0].strip(), parts[1].strip()
                        if key.startswith("Profundidad Relativa"): data_espacial.append([key, value])
                        elif current_section == 'Rendimiento': data_rendimiento.append([key, value])
                        elif current_section == 'Espacial': data_espacial.append([key, value])
                        elif current_section == 'Audio': data_audio.append([key, value])
                        else: data_rendimiento.append([key, value])
        except Exception as e_read:
            print(f"Error leyendo/parseando métricas: {e_read}")
            story.append(Paragraph(f"Error al leer/parsear archivo de métricas: {e_read}", styles['Italic']))
        table_style = TableStyle([('BACKGROUND', (0,0), (-1,0), colors.darkblue), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'LEFT'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTNAME', (0,1), (-1,-1), 'Helvetica'), ('FONTSIZE', (0,0), (-1,-1), 9), ('BOTTOMPADDING', (0,0), (-1,0), 8), ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,1), (-1,-1), 3), ('TEXTCOLOR', (0,1), (-1,-1), colors.black), ('GRID', (0,0), (-1,-1), 0.5, colors.darkgrey),])
        col_widths = [2.8*inch, 3.2*inch]
        if len(data_rendimiento) > 1: story.append(Paragraph("Rendimiento y Tracking", styles['h3'])); story.append(Spacer(1, 0.1*inch)); table_r = Table(data_rendimiento, colWidths=col_widths); table_r.setStyle(table_style); story.append(table_r); story.append(Spacer(1, 0.2*inch))
        if len(data_espacial) > 1: story.append(Paragraph("Análisis Espacial y Movimiento", styles['h3'])); story.append(Spacer(1, 0.1*inch)); table_e = Table(data_espacial, colWidths=col_widths); table_e.setStyle(table_style); story.append(table_e); story.append(Spacer(1, 0.2*inch))
        if selected_transcript and len(data_audio) > 1: story.append(Paragraph("Análisis de Audio", styles['h3'])); story.append(Spacer(1, 0.1*inch)); table_a = Table(data_audio, colWidths=col_widths); table_a.setStyle(table_style); story.append(table_a); story.append(Spacer(1, 0.2*inch))
        page_width, page_height = letter; available_width = page_width - 1.5*inch; available_height = page_height - 2*inch
        def add_image_to_story(image_path, caption_text):
            try:
                with PILImage.open(image_path) as img: img_width, img_height = img.size; aspect = img_height / float(img_width); new_width = available_width; new_height = new_width * aspect
                if new_height > available_height * 0.8: new_height = available_height * 0.8; new_width = new_height / aspect
                img_reportlab = ReportlabImage(image_path, width=new_width, height=new_height); story.append(PageBreak()); story.append(Paragraph(caption_text, styles['h3'])); story.append(Spacer(1, 0.1*inch)); story.append(img_reportlab); story.append(Spacer(1, 0.2*inch))
            except Exception as e_img: story.append(Paragraph(f"Error al cargar imagen {os.path.basename(image_path)}: {e_img}", styles['Italic']))
        add_image_to_story(ruta_pos, "Mapa de Calor de Posición"); add_image_to_story(ruta_traj, "Trayectoria del Docente")
        doc.build(story);
        if messagebox.askyesno("Éxito", f"Reporte PDF generado en:\n{pdf_save_path}\n\n¿Deseas abrirlo ahora?"): abrir_archivo(pdf_save_path)
    except Exception as e_pdf: messagebox.showerror("Error PDF", f"Error al generar PDF:\n{e_pdf}")

def abrir_ventana_configuracion():
    config_window = tk.Toplevel(root)
    config_window.title("Configuración")
    config_window.geometry("550x450")
    config_window.resizable(False, False)
    config_window.configure(bg="#f0f0f0")
    config_window.grab_set()
    config_window.transient(root)
    main_config_frame = ttk.Frame(config_window, padding="15 15 15 15")
    main_config_frame.pack(expand=True, fill=tk.BOTH)
    config = configparser.ConfigParser()
    config.read('config.ini')
    vars = {}

    def guardar_configuracion():
        try:
            for section, fields in entries.items():
                for key, var in fields.items():
                    config.set(section, key, var.get())
            with open('config.ini', 'w') as configfile:
                config.write(configfile)
            messagebox.showinfo("Éxito", "Configuración guardada correctamente.", parent=config_window)
            config_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la configuración:\n{e}", parent=config_window)

    entries = {'Paths': {}, 'Models': {}, 'Parameters': {}}
    
    paths_frame = ttk.LabelFrame(main_config_frame, text=" Rutas (Paths) ", padding="10")
    paths_frame.pack(fill="x", expand=True, pady=5)
    paths_frame.columnconfigure(1, weight=1)
    fields_paths = ['output_base_dir', 'default_video_dir', 'default_transcript_dir']
    for i, field in enumerate(fields_paths):
        ttk.Label(paths_frame, text=f"{field}:").grid(row=i, column=0, sticky="w", pady=2, padx=5)
        vars[field] = tk.StringVar(value=config.get('Paths', field, fallback=''))
        ttk.Entry(paths_frame, textvariable=vars[field]).grid(row=i, column=1, sticky="ew", pady=2)
        entries['Paths'][field] = vars[field]

    models_frame = ttk.LabelFrame(main_config_frame, text=" Modelos (Models) ", padding="10")
    models_frame.pack(fill="x", expand=True, pady=5)
    models_frame.columnconfigure(1, weight=1)
    fields_models = ['yolo_model', 'depth_model']
    for i, field in enumerate(fields_models):
        ttk.Label(models_frame, text=f"{field}:").grid(row=i, column=0, sticky="w", pady=2, padx=5)
        vars[field] = tk.StringVar(value=config.get('Models', field, fallback=''))
        ttk.Entry(models_frame, textvariable=vars[field]).grid(row=i, column=1, sticky="ew", pady=2)
        entries['Models'][field] = vars[field]
        
    params_frame = ttk.LabelFrame(main_config_frame, text=" Parámetros (Parameters) ", padding="10")
    params_frame.pack(fill="x", expand=True, pady=5)
    params_frame.columnconfigure(1, weight=1)
    fields_params = ['yolo_confidence', 'iou_threshold', 'proximity_threshold_px', 'stillness_threshold_px']
    for i, field in enumerate(fields_params):
        ttk.Label(params_frame, text=f"{field}:").grid(row=i, column=0, sticky="w", pady=2, padx=5)
        vars[field] = tk.StringVar(value=config.get('Parameters', field, fallback=''))
        ttk.Entry(params_frame, textvariable=vars[field]).grid(row=i, column=1, sticky="ew", pady=2)
        entries['Parameters'][field] = vars[field]
    
    button_frame = ttk.Frame(main_config_frame)
    button_frame.pack(fill="x", expand=False, pady=(20, 0))
    save_button = ttk.Button(button_frame, text="💾 Guardar y Cerrar", command=guardar_configuracion, style="DarkBlue.TButton")
    save_button.pack(side="right", padx=5)
    cancel_button = ttk.Button(button_frame, text="❌ Cancelar", command=config_window.destroy)
    cancel_button.pack(side="right")


# --- Crear ventana principal y widgets ---
root = tk.Tk(); root.title("📹 Procesador de Videos Docentes v2.5"); root.geometry("550x720")
root.resizable(True, True) # <-- LÍNEA MODIFICADA
style = ttk.Style(); style.theme_use('clam')

style.configure('DarkBlue.TButton', font=("Segoe UI", 10, "bold"), padding=6, background='#043c50', foreground='white')
style.map('DarkBlue.TButton', background=[('pressed', '#032836'), ('active', '#054e66'), ('disabled', '#6c757d')], foreground=[('disabled', '#adb5bd')])
style.configure("TLabel", font=("Segoe UI", 10), background="#f0f0f0", foreground="#333")
style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), background="#f0f0f0", foreground="#005f99"); style.configure("TLabelframe", background="#f0f0f0")
style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), background="#f0f0f0", foreground="#333"); style.configure("Horizontal.TScale", background="#f0f0f0")
main_frame = ttk.Frame(root, padding="20 20 20 20"); main_frame.pack(expand=True, fill=tk.BOTH)
title_label = ttk.Label(main_frame, text="Analizador de Movimiento Docente", style="Header.TLabel", anchor="center"); title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15))
video_frame = ttk.LabelFrame(main_frame, text=" 1. Archivo de Video ", padding="10 5 10 5"); video_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 5)); video_frame.columnconfigure(1, weight=1)

select_button = ttk.Button(video_frame, text="📂 Seleccionar Video", command=seleccionar_video, style='DarkBlue.TButton')
select_button.grid(row=0, column=0, padx=(0, 10))
video_label = ttk.Label(video_frame, text="No se ha seleccionado video", width=40, anchor="w"); video_label.grid(row=0, column=1, sticky="ew")
transcript_frame = ttk.LabelFrame(main_frame, text=" 2. Transcripción (Opcional) ", padding="10 5 10 5"); transcript_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 10)); transcript_frame.columnconfigure(1, weight=1)
transcript_button = ttk.Button(transcript_frame, text="📜 Seleccionar CSV", command=seleccionar_transcripcion, style='DarkBlue.TButton')
transcript_button.grid(row=0, column=0, padx=(0, 10))
transcript_label = ttk.Label(transcript_frame, text="Ninguna transcripción seleccionada.", width=40, anchor="w"); transcript_label.grid(row=0, column=1, sticky="ew")
time_frame = ttk.LabelFrame(main_frame, text=" 3. Intervalo a Procesar ", padding="10 5 10 5"); time_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 10)); time_frame.columnconfigure(1, weight=1); time_frame.columnconfigure(2, weight=0)
start_label = ttk.Label(time_frame, text="Inicio (MM:SS):"); start_label.grid(row=0, column=0, padx=(0, 5), sticky="w"); slider_start = ttk.Scale(time_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=validar_intervalo, length=300); slider_start.grid(row=0, column=1, padx=5, sticky="ew")
start_label_val = ttk.Label(time_frame, text="00:00", width=6, anchor="e"); start_label_val.grid(row=0, column=2, padx=(5, 0), sticky="e")
end_label = ttk.Label(time_frame, text="Fin (MM:SS):"); end_label.grid(row=1, column=0, padx=(0, 5), sticky="w", pady=(5,0)); slider_end = ttk.Scale(time_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=validar_intervalo, length=300); slider_end.grid(row=1, column=1, padx=5, sticky="ew", pady=(5,0))
end_label_val = ttk.Label(time_frame, text="00:00", width=6, anchor="e"); end_label_val.grid(row=1, column=2, padx=(5, 0), sticky="e", pady=(5,0))

options_frame = ttk.LabelFrame(main_frame, text=" 4. Opciones de Procesamiento ", padding="10 5 10 5")
options_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 15))
options_frame.columnconfigure(1, weight=1)
frame_skip_label = ttk.Label(options_frame, text="Nivel de Salto de Frames (0=ninguno):")
frame_skip_label.grid(row=0, column=0, padx=(0, 5), sticky="w")
frame_skip_var = tk.DoubleVar(value=0)
frame_skip_slider = ttk.Scale(options_frame, from_=0, to=10, orient=tk.HORIZONTAL, variable=frame_skip_var, command=update_frame_skip_label)
frame_skip_slider.bind("<B1-Motion>", lambda e: frame_skip_var.set(round(frame_skip_var.get())))
frame_skip_slider.grid(row=0, column=1, padx=5, sticky="ew")
frame_skip_val_label = ttk.Label(options_frame, text="0", width=3, anchor="e")
frame_skip_val_label.grid(row=0, column=2, padx=(5, 0), sticky="e")

action_frame = ttk.Frame(main_frame); action_frame.grid(row=5, column=0, columnspan=3, pady=(5, 5))
config_button = ttk.Button(action_frame, text="⚙️ Configuración", command=abrir_ventana_configuracion, style='DarkBlue.TButton')
config_button.pack(side=tk.LEFT, padx=5)
process_button = ttk.Button(action_frame, text="🚀 Procesar Video", command=procesar_video_seleccionado, state=tk.DISABLED, style='DarkBlue.TButton')
process_button.pack(side=tk.LEFT, padx=5)
pdf_button = ttk.Button(action_frame, text="📊 Generar PDF", command=generar_pdf, state=tk.DISABLED, style='DarkBlue.TButton')
pdf_button.pack(side=tk.LEFT, padx=5)
pdf_open_button = ttk.Button(action_frame, text="📄 Abrir PDF", command=abrir_pdf_reporte, state=tk.DISABLED, style='DarkBlue.TButton')
pdf_open_button.pack(side=tk.LEFT, padx=5)

history_frame = ttk.LabelFrame(main_frame, text=" Historial de Procesamientos (Doble Clic para abrir carpeta) ", padding="10 5 10 5")
history_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(10, 5))
history_frame.columnconfigure(0, weight=1); history_frame.rowconfigure(0, weight=1)
columns = ('run', 'date', 'time')
history_tree = ttk.Treeview(history_frame, columns=columns, show='headings') # Altura eliminada para que se expanda
history_tree.heading('run', text='Ejecución'); history_tree.heading('date', text='Fecha'); history_tree.heading('time', text='Hora')
history_tree.column('run', width=150, anchor=tk.W); history_tree.column('date', width=100, anchor=tk.CENTER); history_tree.column('time', width=100, anchor=tk.CENTER)
history_tree.grid(row=0, column=0, sticky="nsew")
scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=history_tree.yview); scrollbar.grid(row=0, column=1, sticky="ns")
history_tree.config(yscrollcommand=scrollbar.set)
history_tree.bind("<Double-Button-1>", abrir_carpeta_historial)

status_label = ttk.Label(main_frame, text="Selecciona un video para empezar", anchor="center"); status_label.grid(row=7, column=0, columnspan=3, pady=(10, 0))

# --- CONFIGURACIÓN DE EXPANSIÓN ---
main_frame.rowconfigure(6, weight=1) 
main_frame.columnconfigure(0, weight=1)
# ---------------------------------

style.configure("Treeview", rowheight=25)
history_tree.tag_configure('oddrow', background='#E8E8E8')
history_tree.tag_configure('evenrow', background='white')

if not reportlab_available:
    pdf_button.config(state=tk.DISABLED)
    pdf_open_button.config(state=tk.DISABLED)
    status_label.config(text="Advertencia: ReportLab/Pillow no instalados. PDF desactivado.")

root.mainloop()