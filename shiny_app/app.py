import os
import sys
import cv2
import glob
import paramiko
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pkg_resources
import signal
import time

# Shiny Imports
from shiny import reactive, render, ui
from shiny.express import input, output, ui

# --- 1. PATHING & IMPORTS ---
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

# Resolve the package data directory and ensure WEIGHTS_DIR is a Path object
package_data_path = pkg_resources.resource_filename('berryboxai', 'data')
WEIGHTS_DIR = Path(package_data_path) / 'weights'

sys.path.insert(0, str(APP_DIR))

from utils.style import LIGHT_CSS
from utils.helpers import (
    annotated_image_rgb, 
    load_model, 
    build_model_params, 
    setup_nikon_camera
)

# Import your custom feature extractors
from utils.functions import color_correction, read_QR_code, get_all_features_parallel, get_ids
from utils.helpers import summarize_rot_det_results # The new one we just added

# --- 2. GLOBAL REACTIVE STATE ---
ssh_client = reactive.Value(None)
camera_ok = reactive.Value(False)
log_lines = reactive.Value(["[System] Ready."])
last_processed_img = reactive.Value(None)
last_metrics = reactive.Value({})
batch_results = reactive.Value(None) # Added this to prevent errors in Batch tab

def add_log(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    tag = {"info":"·","ok":"✓","warn":"⚠","err":"✗"}.get(level,"·")
    current_logs = log_lines.get()
    new_logs = current_logs + [f"[{ts}] {tag} {msg}"]
    log_lines.set(new_logs)

# --- 3. MODEL LOADER ---
@reactive.calc
def get_model():
    module_name = input.module()
    task = "segment" if module_name == "berry-seg" else "detect"
    
    # Pre-check for OpenVINO conversion to warn the user
    # OpenVINO should be used for Windows or macOS with x86_64 architecture, but we check at runtime to be safe
    import platform
    # Is the system Windows or macOS with x86_64 architecture?
    is_ov_eligible = (platform.system() == "Windows") or (platform.system() == "Darwin" and platform.machine() == "x86_64")
    
    # Using / with Path object is now safe
    ov_path = WEIGHTS_DIR / f"berrybox_{module_name}_openvino_model"
    
    if is_ov_eligible and not ov_path.exists():
        add_log("FIRST RUN: Converting to OpenVINO (1-3 mins)...", "warn")
    else:
        add_log(f"Loading {module_name}...", "info")

    # Pass weights_dir as string to the helper
    model = load_model(module_name, task, str(WEIGHTS_DIR))
    add_log(f"Model {module_name} ready.", "ok")
    return model

# --- 4. UI ---
ui.page_opts(title="BerryBox AI", fillable=True)
ui.head_content(ui.tags.style(LIGHT_CSS))

with ui.sidebar():
    ui.markdown("## BerryBox AI")
    ui.input_select("module", "Module", {"berry-seg": "Segmentation", "rot-det": "Rot Detection"})
    
    ui.input_slider("conf", "Confidence", 0.1, 1.0, 0.5, step=0.05)
    ui.input_slider("iou", "IoU", 0.1, 0.9, 0.25, step=0.05)

    ui.hr()
    ui.markdown("### Data Export (Required)")
    with ui.layout_column_wrap(width=1, fill=False):
        ui.input_text("save_base_dir", "Output Folder Path", placeholder="C:/BerryBox/Data")
        # Added the browse button here
        ui.input_action_button("btn_browse", "📂 Browse Folder", class_="btn-outline-secondary btn-sm")
    
    ui.input_text("session_name", "Session ID", value=datetime.now().strftime("%Y%m%d_%H%M"))
    
    ui.hr()
    ui.markdown("### Image Settings")
    ui.input_checkbox("do_cc", "Enable Color Correction", value=True)
    ui.input_checkbox("do_qr", "Read QR/Barcode", value=True)
    ui.input_numeric("patch_size", "CC Patch Size (cm)", value=1.2)
    
    ui.hr()
    ui.markdown("### Raspberry Pi")
    ui.input_text("rpi_ip", "IP Address", "169.254.111.10")
    ui.input_text("rpi_user", "Username", "cranpi2") 
    ui.input_password("rpi_pwd", "Password", value="usdacran")


with ui.navset_bar(title="BerryBox AI"):
    
    with ui.nav_panel("📷 Interactive"):
        with ui.layout_columns(col_widths=[4, 8]):
            with ui.card():
                ui.card_header("System Control")
                ui.input_action_button("btn_connect", "🔌 Connect Pi", class_="btn-primary")
                ui.input_action_button("btn_check_cam", "📷 Setup Nikon")
                ui.hr()
                
                # --- NEW SAMPLE ID INPUT ---
                ui.input_text(
                    "sample_id", 
                    "Sample ID (Scan QR or Type)", 
                    placeholder="e.g., Plot_101",
                    width="100%"
                )
                
                ui.input_action_button(
                    "btn_capture", 
                    "Capture and Analyze", 
                    class_="btn-success btn-lg w-100"
                )
                
                ui.div(
                    ui.input_action_button(
                        "btn_shutdown", 
                        "Shutdown App", 
                        class_="btn-danger btn-lg w-100"
                    ),
                    style="margin-top: 10px;"
                )
                
                @render.text
                def display_logs():
                    return "\n".join(log_lines.get()[-12:])

            with ui.layout_columns(col_widths=[12, 12]):
                with ui.card():
                    ui.card_header("Last Capture Preview")
                    @render.image
                    def interactive_preview():
                        img = last_processed_img()
                        if img is None: 
                            return None
                        
                        temp_path = APP_DIR / "temp_interactive.jpg"
                        cv2.imwrite(str(temp_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        
                        # Added CSS styles to force-fit the image to the UI card
                        return {
                            "src": str(temp_path),
                            "style": "width: 100%; max-height: 75vh; object-fit: contain;"
                        }

                with ui.layout_columns():
                    @render.ui
                    def metric_berries():
                        m = last_metrics()
                        return ui.value_box(title="Total Berries", value=str(m.get("total", 0)), theme="primary")
                    
                    @render.ui
                    def metric_rot():
                        m = last_metrics()
                        val = f"{m.get('pct_rot', 0):.1f}%"
                        return ui.value_box(title="Rot Percentage", value=val, theme="danger")

    with ui.nav_panel("📁 Batch"):
        with ui.card():
            ui.input_text("batch_path", "Folder Path", placeholder="C:/path/to/images")
            ui.input_action_button("btn_run_batch", "▶ Run Analysis", class_="btn-success")
        
        with ui.card():
            @render.data_frame
            def batch_table():
                df = batch_results.get()
                if df is not None:
                    return render.DataTable(df)

# --- 5. SERVER LOGIC ---

@reactive.effect
@reactive.event(input.btn_browse)
def _browse_folder():
    import tkinter as tk
    from tkinter import filedialog
    
    # Initialize tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()
    # Bring the dialog to the front of other windows
    root.attributes("-topmost", True)
    
    selected_dir = filedialog.askdirectory(title="Select Output Folder")
    root.destroy()
    
    if selected_dir:
        # Normalize path for Windows compatibility
        normalized_path = os.path.normpath(selected_dir).replace("\\", "/")
        ui.update_text("save_base_dir", value=normalized_path)

@reactive.effect
@reactive.event(input.btn_shutdown)
def _shutdown_app():
    add_log("Shutting down...", "warn")
    
    # JavaScript to try and close the browser tab
    ui.insert_ui(
        ui.tags.script("window.close();"),
        selector="body",
        where="beforeEnd"
    )
    
    # Small delay to allow log/JS to process
    import time
    import signal
    time.sleep(0.5)
    os.kill(os.getpid(), signal.SIGTERM)

@reactive.effect
@reactive.event(input.btn_shutdown)
def _shutdown_app():
    add_log("Initiating shutdown sequence...", "warn")
    
    # 1. Attempt to close the browser window via JavaScript
    ui.insert_ui(
        ui.tags.script("setTimeout(function() { window.close(); }, 500);"),
        selector="body",
        where="beforeEnd"
    )
    
    # 2. Wait a moment to let the JS reach the browser, then kill the backend
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGTERM) 

@reactive.effect
@reactive.event(input.btn_browse)
def _browse_folder():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes("-topmost", True)  # Bring dialog to front
    selected_dir = filedialog.askdirectory()
    root.destroy()
    if selected_dir:
        ui.update_text("save_base_dir", value=selected_dir)

@reactive.effect
@reactive.event(input.btn_connect)
def _connect_pi():
    add_log(f"Connecting to {input.rpi_ip()}...", "info")
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            input.rpi_ip(), 
            username=input.rpi_user(), 
            password=input.rpi_pwd(), 
            timeout=8
        )
        ssh_client.set(client)
        add_log("SSH connection established.", "ok")
    except Exception as e:
        add_log(f"Connection failed: {str(e)}", "err")

@reactive.effect
@reactive.event(input.btn_check_cam)
def _check_cam():
    client = ssh_client.get()
    if not client:
        add_log("Cannot setup: SSH not connected.", "warn")
        return
    try:
        add_log("Initializing Nikon D7500 settings...", "info")
        msg = setup_nikon_camera(client)
        camera_ok.set(True)
        add_log(msg, "ok")
    except Exception as e:
        camera_ok.set(False)
        add_log(f"Nikon error: {str(e)}", "err")

@reactive.effect
@reactive.event(input.btn_capture)
def _handle_capture():
    # --- 1. NEW: SAMPLE ID CHECK ---
    current_sample_id = input.sample_id().strip()
    if not current_sample_id:
        add_log("Sample ID is required! Please scan or type it.", "err")
        return

    # --- PRE-FLIGHT CHECKS ---
    if not input.save_base_dir() or not os.path.exists(input.save_base_dir()):
        add_log("Output folder is missing or invalid! Set it first.", "err")
        return
        
    client = ssh_client.get()
    if not client:
        add_log("Connect SSH first!", "err")
        return
    if not camera_ok.get():
        add_log("Setup Nikon first!", "warn")
        return

    try:
        current_model = get_model()
        
        # Setup directories
        export_path = Path(input.save_base_dir()) / input.session_name()
        img_export_path = export_path / "images"
        img_export_path.mkdir(parents=True, exist_ok=True)

        with ui.Progress(min=1, max=5) as p:
            # STEP 1: Capture
            p.set(1, message="Nikon: Triggering Shutter...")
            remote_img = "/tmp/capture.jpg"
            cmd = f"gphoto2 --capture-image-and-download --filename {remote_img} --force-overwrite"
            stdin, stdout, stderr = client.exec_command(cmd)
            
            if stdout.channel.recv_exit_status() != 0:
                add_log("Nikon Capture Failed!", "err")
                return

            # STEP 2: Download
            p.set(2, message="Downloading from Pi...")
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Include the sample ID in the image name
            image_name = f"{current_sample_id}_{current_datetime}.jpg"
            local_raw = str(img_export_path / image_name)
            
            sftp = client.open_sftp()
            sftp.get(remote_img, local_raw)
            sftp.close()

            # STEP 3: Inference
            p.set(3, message="AI: Running YOLO...")
            img_bgr = cv2.imread(local_raw)
            h, w = img_bgr.shape[:2]

            cfg = {"module": input.module(), "conf": input.conf(), "iou": input.iou(), "imgsz": (h, w)}
            params = build_model_params(cfg)
            results = current_model.predict(img_bgr, **params)
            result = results[0].to("cpu")

            # STEP 4: Feature Extraction
            p.set(4, message="Extracting Features...")
            csv_path = export_path / f"{input.session_name()}_features.csv"
            
            total_objs = 0
            pct_rot = 0

            if input.module() == "berry-seg":
                # Color Correction
                cc_patch_size = 0
                if input.do_cc():
                    try:
                        result, cc_patch_sizes = color_correction(result)
                        cc_patch_size = np.min(cc_patch_sizes)
                    except Exception as e:
                        add_log(f"Color Correction skipped: {e}", "warn")
                
                # QR Code
                barcode = image_name
                if input.do_qr() and any(result.boxes.cls == get_ids(result, 'info')[0]):
                    barcode = read_QR_code(result)

                # Get Deep Features
                df1 = get_all_features_parallel(result, name='berry')
                df2 = get_all_features_parallel(result, name='rotten')
                
                df_features = pd.concat([
                    pd.DataFrame({'name': (['berry'] * df1.shape[0]) + (['rotten'] * df2.shape[0])}), 
                    pd.concat([df1, df2], ignore_index=True)
                ], axis=1)
                
                total_objs = df_features.shape[0]
                
                if total_objs > 0:
                    # Construct Metadata Front
                    df_meta = pd.DataFrame({
                        'Date': [current_datetime] * total_objs,
                        'Image Name': [image_name] * total_objs,
                        'Sample_ID': [current_sample_id] * total_objs,
                        'QR_info': [barcode] * total_objs,
                        'Object_ID': list(range(total_objs)),
                        'Patch_size': [cc_patch_size] * total_objs
                    })
                    
                    df_final = pd.concat([df_meta, df_features], axis=1)
                    
                    # Compute Physics
                    df_final["Ellipsoid_model_volume"] = (4/3) * np.pi * (df_final["RP_Minor_axis_length"] / 2) * ((df_final["RP_Major_axis_length"] / 2) ** 2)
                    df_final["Eccentricity"] = np.sqrt(1 - ((0.5 * df_final["RP_Major_axis_length"]) ** 2 / (0.5 * df_final["RP_Minor_axis_length"]) ** 2))
                    
                    # Convert to CM based on patch size
                    if cc_patch_size > 0:
                        cm_per_pixel = float(input.patch_size()) / cc_patch_size
                        df_final["Area"] = df_final["RP_Area"] * (cm_per_pixel ** 2)
                        df_final["Length"] = df_final["RP_Minor_axis_length"] * cm_per_pixel
                        df_final["Width"] = df_final["RP_Major_axis_length"] * cm_per_pixel
                        df_final["Ellipsoid_model_volume"] = df_final["Ellipsoid_model_volume"] * (cm_per_pixel ** 3)
                        df_final["cm_per_pixel"] = cm_per_pixel

                    # Save to CSV
                    if csv_path.exists():
                        existing_df = pd.read_csv(csv_path)
                        df_final = pd.concat([existing_df, df_final], ignore_index=True)
                    df_final.to_csv(csv_path, index=False)

            elif input.module() == "rot-det":
                objects_count, n_rotten, n_sound, perc_rot, weighted_perc_rot = summarize_rot_det_results(result)
                
                if objects_count > 0:
                    df_final = pd.DataFrame({
                        'Date': [current_datetime],
                        'Image Name': [image_name],
                        'Sample_ID': [current_sample_id],
                        'NumberSoundBerries': [n_sound],
                        'NumberRottenBerries': [n_rotten],
                        'FruitRotPer': [perc_rot],
                        'FruitRotPerWtd': [weighted_perc_rot]
                    })
                    if csv_path.exists():
                        existing_df = pd.read_csv(csv_path)
                        df_final = pd.concat([existing_df, df_final], ignore_index=True)
                    df_final.to_csv(csv_path, index=False)

            # STEP 5: Update UI Metrics (Filtering out non-fruit objects)
            p.set(5, message="Finalizing Results...")
            class_names = ["berry", "rotten", "sound"]
            annotated = annotated_image_rgb(img_bgr, result, class_names)

            # Map the valid fruit IDs
            valid_fruit_names = {"berry", "rotten", "sound"}
            valid_ids = [k for k, v in result.names.items() if v in valid_fruit_names]
            rot_ids = [k for k, v in result.names.items() if v == "rotten"]
            detected_classes = result.boxes.cls.cpu().numpy()
            
            # Recalculate metrics just for fruit
            total_berries = sum(1 for c in detected_classes if c in valid_ids)
            n_rotten_ui = sum(1 for c in detected_classes if c in rot_ids)
            ui_pct_rot = (n_rotten_ui / total_berries * 100) if total_berries > 0 else 0

            # Update UI state
            last_processed_img.set(annotated)
            last_metrics.set({"total": total_berries, "pct_rot": ui_pct_rot})
            
            # Clear the Sample ID box for the next scan
            ui.update_text("sample_id", value="")
            
            add_log(f"Processed & Saved {total_berries} items to CSV.", "ok")

    except Exception as e:
        add_log(f"Error: {str(e)}", "err")

@reactive.effect
@reactive.event(input.btn_run_batch)
def _handle_batch():
    # Placeholder logic for batch processing
    add_log("Batch processing not yet fully implemented in this server block.", "info")
    pass