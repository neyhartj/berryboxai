import os
import datetime
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import re
import shutil
import cv2
from shiny import App, ui, reactive, render

# Import custom package classes
from cran_field_therm.field_book import FieldBook
from cran_field_therm.camera_control import CameraController

def get_directory_path(title="Select Folder"):
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder_path = filedialog.askdirectory(parent=root, title=title)
    root.destroy()
    return folder_path

book = FieldBook()
camera = CameraController()
default_docs_path = os.path.join(os.path.expanduser("~"), "Documents")

# --- UI DEFINITION ---
app_ui = ui.page_navbar(
    
    # ---------------------------------------------------------
    # TAB 1: MAIN CAPTURE APP
    # ---------------------------------------------------------
    ui.nav_panel(
        "Capture App",
        ui.navset_hidden(
            # SCREEN 1: Configuration & Validation
            ui.nav_panel(
                "screen1",
                ui.h2("Setup FLIR Capture Session"),
                ui.card(
                    ui.input_file("csv_file", "1. Upload Field Book (.csv)", accept=[".csv"]),
                    ui.output_text("validation_status"),
                    ui.br(),
                    
                    ui.output_ui("traversal_ui"),
                    
                    ui.input_numeric("num_replicates", "3. Replicate Images per Plot", value=10, min=1),
                    
                    ui.h5("4. Destination Folder"),
                    ui.row(
                        ui.column(9, ui.input_text("dest_folder", "", value=default_docs_path, width="100%")),
                        ui.column(3, ui.input_action_button("browse_btn", "📁 Browse...", class_="btn-secondary w-100"))
                    ),
                    ui.br(),
                    
                    ui.input_action_button("connect_btn", "Validate & Connect Camera", class_="btn-primary")
                )
            ),
            
            # SCREEN 2: Capture Interface
            ui.nav_panel(
                "screen2",
                ui.h2("Field Capture Mode"),
                ui.card(
                    # Top Bar: Identification Context
                    ui.h3(ui.output_text("current_plot_display")),
                    # Top Bar: Sequential Flow Control Buttons
                    ui.row(
                        ui.column(6, ui.input_action_button("prev_btn", "⬅ Previous Plot", class_="w-100")),
                        ui.column(6, ui.input_action_button("next_btn", "Next Plot ➡", class_="w-100"))
                    ),
                    ui.hr(),
                    
                    # Jump To Navigation
                    ui.row(
                        ui.column(4, ui.input_select("jump_field", "Jump To:", {"unique_id": "Unique ID", "plot": "Plot", "row": "Row", "column": "Column"})),
                        ui.column(5, ui.input_text("jump_value", "Search Value:")),
                        ui.column(3, ui.div(ui.input_action_button("jump_btn", "Go 🚀", class_="btn-info w-100"), style="margin-top: 30px;"))
                    ),
                    ui.hr(),
                    
                    # Dynamic Layout Split
                    ui.row(
                        # Left Sidebar Columns (Width 4/12)
                        ui.column(
                            4,
                            ui.input_action_button("capture_btn", "📸 Capture Thermal Data", class_="btn-danger btn-lg w-100"),
                            
                            ui.div(
                                ui.output_text_verbatim("capture_status_output"),
                                style="margin-top: 15px; font-weight: bold;"
                            ),
                            
                            ui.hr(),
                            
                            # Completeness Tracker Container
                            ui.div(
                                ui.output_ui("completeness_tracker"),
                                style="margin-top: 10px; margin-bottom: 15px;"
                            ),
                            
                            ui.div(style="height: 30px;"), 
                            
                            ui.input_action_button("shutdown_btn", "🛑 Shutdown App", class_="btn-dark w-100")
                        ),
                        
                        # Right Image Window Column (Width 8/12)
                        ui.column(
                            8,
                            ui.div(
                                ui.output_ui("captured_preview"),
                                style="height: 450px; display: flex; align-items: center; justify-content: center; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px;"
                            )
                        )
                    )
                )
            ),
            id="wizard"
        )
    ),
    
    # ---------------------------------------------------------
    # TAB 2: ANALYZE & CONSOLIDATE DATA
    # ---------------------------------------------------------
    ui.nav_panel(
        "Analyze Data",
        ui.h2("Data Consolidation & Re-Extraction"),
        ui.p("Use this tool to clean up your data if you had to restart sessions, accidentally took too many images, or need to extract fresh temperature metrics from your raw TIFF files."),
        ui.card(
            ui.h4("1. Original Field Book"),
            ui.input_file("analysis_csv_file", "Upload Original Field Book (.csv)", accept=[".csv"]),
            
            ui.h4("2. Parent Folder Containing Sessions"),
            ui.p("Place all of the raw image session folders you want to analyze into a single main folder. Select that main folder below.", style="color: #6c757d; font-size: 0.9em;"),
            ui.row(
                ui.column(9, ui.input_text("analysis_parent_folder", "", value="", width="100%")),
                ui.column(3, ui.input_action_button("browse_parent_btn", "📁 Browse...", class_="btn-secondary w-100"))
            ),
            ui.br(),
            
            ui.h4("3. Extraction Settings"),
            ui.input_numeric("analysis_num_replicates", "Target Images per Plot (n_i)", value=10, min=1),
            
            ui.h4("4. Final Output Destination"),
            ui.row(
                ui.column(9, ui.input_text("analysis_dest_folder", "", value=default_docs_path, width="100%")),
                ui.column(3, ui.input_action_button("analysis_browse_dest_btn", "📁 Browse...", class_="btn-secondary w-100"))
            ),
            ui.br(),
            
            ui.input_action_button("run_analysis_btn", "🚀 Run Analysis & Consolidation", class_="btn-success btn-lg w-100"),
            ui.br(),
            
            ui.output_text_verbatim("analysis_log")
        )
    ),

    # ---------------------------------------------------------
    # TAB 3: TROUBLESHOOTING
    # ---------------------------------------------------------
    ui.nav_panel(
        "Troubleshooting",
        ui.card(
            ui.h3("Troubleshooting Guide"),
            ui.p("Problems running the app? Try the following:"),
            ui.tags.ol(
                ui.tags.li("Restart the laptop")
            )
        )
    ),
    
    title="CranFieldTherm"
)

# --- SERVER LOGIC ---
def server(input, output, session):
    
    # ---------------------------------------------------------
    # CAPTURE APP VARIABLES
    # ---------------------------------------------------------
    current_index = reactive.Value(0)
    csv_valid = reactive.Value(False)
    session_save_path = reactive.Value("") 
    session_csv_path = reactive.Value("")
    latest_preview_b64 = reactive.Value(None)
    completed_plots_count = reactive.Value(0)
    capture_status_msg = reactive.Value("Ready.")
    capture_trigger = reactive.Value(0)
    
    # =========================================================
    # CAPTURE APP LOGIC
    # =========================================================
    @output
    @render.text
    def validation_status():
        file_infos = input.csv_file()
        if not file_infos:
            return "Awaiting CSV upload..."
        try:
            book.load_and_validate(file_infos[0]["datapath"])
            csv_valid.set(True)
            msg = "✅ CSV Validated successfully."
            if book.trial_name:
                msg += f" (Trial Detected: {book.trial_name})"
            return msg
        except Exception as e:
            csv_valid.set(False)
            return f"❌ Error: {str(e)}"

    @output
    @render.ui
    def traversal_ui():
        if not csv_valid.get():
            return None
        options = {
            "col_in_row": "Column-within-Row Order",
            "row_in_col": "Row-within-Column Order"
        }
        if book.has_plot:
            options = {"plot": "Plot Order", **options}
            
        return ui.input_radio_buttons("order_type", "2. Select Traversal Order", options)

    @reactive.Effect
    @reactive.event(input.browse_btn)
    def browse_for_folder():
        selected_folder = get_directory_path("Select Capture Output Folder")
        if selected_folder:
            windows_path = os.path.normpath(selected_folder)
            ui.update_text("dest_folder", value=windows_path)

    @reactive.Effect
    @reactive.event(input.connect_btn)
    def handle_connection():
        if not csv_valid.get():
            ui.notification_show("Please upload a valid CSV first.", type="error")
            return
            
        base_dest = input.dest_folder()
        if not os.path.exists(base_dest):
             ui.notification_show("Destination folder does not exist. Please create it.", type="error")
             return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if book.trial_name:
            session_folder_name = f"{book.trial_name}_cran_field_therm_{timestamp}"
        else:
            session_folder_name = f"cran_field_therm_{timestamp}"
            
        session_parent_path = os.path.join(base_dest, session_folder_name)
        raw_images_path = os.path.join(session_parent_path, "raw_images")
        
        try:
            os.makedirs(raw_images_path, exist_ok=True) 
            session_save_path.set(raw_images_path)
            
            csv_out_name = f"{book.trial_name}_field_thermal_data.csv"
            session_csv_path.set(os.path.join(session_parent_path, csv_out_name))
        except Exception as e:
            ui.notification_show(f"Failed to create session folders: {e}", type="error")
            return

        book.sort_data(input.order_type())
        
        try:
            camera.connect()
            ui.notification_show("Camera connected! Session folders created.", type="message")
            
            completed_plots_count.set(0)
            ui.update_navset("wizard", selected="screen2")
            current_index.set(0)
        except Exception as e:
            ui.notification_show(f"Camera Connection Failed: {e}", type="error")

    @output
    @render.text
    def current_plot_display():
        if not csv_valid.get() or book.df is None:
            return ""
        
        idx = current_index.get()
        row_data = book.df.iloc[idx]
        display = f"Unique ID: {row_data['unique_id']} | Row: {row_data['row']} | Col: {row_data['column']}"
        if book.has_plot:
            display += f" | Plot: {row_data['plot']}"
        return display

    @reactive.Effect
    @reactive.event(input.next_btn)
    def next_plot():
        if current_index.get() < len(book.df) - 1:
            current_index.set(current_index.get() + 1)
        else:
            ui.notification_show("End of field book reached.", type="warning")

    @reactive.Effect
    @reactive.event(input.prev_btn)
    def prev_plot():
        if current_index.get() > 0:
            current_index.set(current_index.get() - 1)

    @reactive.Effect
    @reactive.event(input.jump_btn)
    def handle_jump():
        if book.df is None:
            return
            
        field = input.jump_field()
        search_val = input.jump_value().strip()
        
        if not search_val:
            ui.notification_show("Please enter a search value.", type="warning")
            return

        # Convert column to string for robust comparison against text input
        matches = book.df.index[book.df[field].astype(str) == str(search_val)].tolist()
        
        if matches:
            current_index.set(matches[0])
            latest_preview_b64.set(None)
            ui.notification_show(f"Jumped to first match for {field} = {search_val}", type="message")
        else:
            ui.notification_show(f"No match found for {field} = {search_val}", type="error")

    @reactive.Effect
    @reactive.event(input.next_btn, input.prev_btn)
    def clear_preview_on_navigate():
        latest_preview_b64.set(None)
        capture_status_msg.set("Ready.")

    @output
    @render.ui
    def captured_preview():
        b64 = latest_preview_b64.get()
        if not b64:
            return ui.div("Awaiting Capture...", style="color: #6c757d; font-style: italic;")
        
        return ui.img(
            src=f"data:image/jpeg;base64,{b64}", 
            style="max-height: 100%; max-width: 100%; object-fit: contain;"
        )

    @output
    @render.ui
    def completeness_tracker():
        if book.df is None:
            return None
            
        total_plots = len(book.df)
        completed = completed_plots_count.get()
        percentage = (completed / total_plots) * 100 if total_plots > 0 else 0
        
        return ui.div(
            ui.div("Session Completeness", style="font-size: 0.9rem; color: #6c757d; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;"),
            ui.div(f"{completed} / {total_plots} Plots Saved ({percentage:.1f}%)", style="font-size: 1.1rem; font-weight: bold; margin-top: 2px; color: #212529;"),
            ui.div(
                ui.div(style=f"width: {percentage}%; height: 100%; background-color: #198754; transition: width 0.4s ease;"),
                style="width: 100%; height: 10px; background-color: #e9ecef; border-radius: 4px; overflow: hidden; margin-top: 8px; border: 1px solid #dee2e6;"
            )
        )

    @output
    @render.text
    def capture_status_output():
        return capture_status_msg.get()

    @reactive.Effect
    @reactive.event(input.capture_btn)
    def initiate_capture():
        idx = current_index.get()
        row_data = book.df.iloc[idx]
        
        # Check if we already have data for this plot (detect overwrite)
        if pd.notnull(row_data.get('mean_temp_C')):
            m = ui.modal(
                f"Data has already been collected for this plot (Recorded Temp: {row_data['mean_temp_C']:.2f}°C). Do you want to overwrite the existing data?",
                title="Overwrite Warning",
                easy_close=True,
                footer=ui.div(
                    ui.modal_button("Cancel"),
                    ui.input_action_button("confirm_overwrite_btn", "Overwrite Data", class_="btn-danger")
                ),
            )
            ui.modal_show(m)
        else:
            capture_trigger.set(capture_trigger.get() + 1)

    @reactive.Effect
    @reactive.event(input.confirm_overwrite_btn)
    def confirm_overwrite():
        ui.modal_remove()
        capture_trigger.set(capture_trigger.get() + 1)

    @reactive.Effect
    @reactive.event(capture_trigger, ignore_init=True)
    def perform_capture():
        capture_status_msg.set("Capturing images... please wait.")
        
        idx = current_index.get()
        current_id = book.df.iloc[idx]['unique_id']
        dest_img_folder = session_save_path.get() 
        dest_csv_file = session_csv_path.get()
        num_reps = input.num_replicates()
        
        try:
            raw_means, b64_preview = camera.capture_plot_images(current_id, dest_img_folder, num_reps)
            latest_preview_b64.set(b64_preview)
            
            celsius_means = [(val * 0.01) - 273.15 for val in raw_means]
            fahrenheit_means = [(c * 1.8) + 32 for c in celsius_means]
            
            c_mean = np.mean(celsius_means)
            c_sd = np.std(celsius_means, ddof=1) if len(celsius_means) > 1 else 0
            
            f_mean = np.mean(fahrenheit_means)
            f_sd = np.std(fahrenheit_means, ddof=1) if len(fahrenheit_means) > 1 else 0
            
            c_frame = celsius_means[-1]
            f_frame = fahrenheit_means[-1]
            
            book.update_row_data(current_id, c_mean, c_sd, f_mean, f_sd, dest_csv_file)
            
            num_completed = int(book.df['mean_temp_C'].notnull().sum())
            completed_plots_count.set(num_completed)
            
            report = (
                f"✅ Saved {num_reps} images & logged to CSV.\n"
                f"----------------------------\n"
                f"Preview Frame Avg:\n"
                f" > {c_frame:.2f}°C  |  {f_frame:.2f}°F\n\n"
                f"Plot Overall Avg ({num_reps} Reps):\n"
                f" > {c_mean:.2f} ± {c_sd:.2f}°C\n"
                f" > {f_mean:.2f} ± {f_sd:.2f}°F"
            )
            capture_status_msg.set(report)
            
            # Auto-advance to the next plot
            if current_index.get() < len(book.df) - 1:
                current_index.set(current_index.get() + 1)
                latest_preview_b64.set(None)
                capture_status_msg.set("Ready. (Auto-advanced to next plot)")
            else:
                ui.notification_show("End of field book reached.", type="warning")
            
        except Exception as e:
            capture_status_msg.set(f"❌ Capture Failed:\n{e}")

    @reactive.Effect
    @reactive.event(input.shutdown_btn)
    def show_shutdown_modal():
        m = ui.modal(
            "Are you sure you want to shut down the application? This will safely disconnect the camera and close the web server.",
            title="Confirm Shutdown",
            easy_close=True,
            footer=ui.div(
                ui.modal_button("Cancel"),
                ui.input_action_button("confirm_shutdown", "Confirm Shutdown", class_="btn-danger")
            ),
        )
        ui.modal_show(m)

    @reactive.Effect
    @reactive.event(input.confirm_shutdown)
    def perform_shutdown():
        ui.modal_remove()
        camera.disconnect()
        os._exit(0)

    session.on_ended(camera.disconnect)

    # =========================================================
    # ANALYZE APP LOGIC
    # =========================================================
    @reactive.Effect
    @reactive.event(input.browse_parent_btn)
    def browse_parent_folder():
        folder = get_directory_path("Select Parent Folder Containing Sessions")
        if folder:
            ui.update_text("analysis_parent_folder", value=os.path.normpath(folder))

    @reactive.Effect
    @reactive.event(input.analysis_browse_dest_btn)
    def browse_analysis_dest():
        folder = get_directory_path("Select Analysis Output Destination")
        if folder:
            ui.update_text("analysis_dest_folder", value=os.path.normpath(folder))

    @output
    @render.text
    @reactive.event(input.run_analysis_btn)
    def analysis_log():
        # Validate Inputs
        file_infos = input.analysis_csv_file()
        if not file_infos:
            return "Error: Please upload an Original Field Book CSV."
            
        parent_dir = input.analysis_parent_folder()
        if not parent_dir or not os.path.exists(parent_dir):
            return "Error: Please select a valid parent folder containing your session folders."
            
        n_i = input.analysis_num_replicates()
        dest_base = input.analysis_dest_folder()
        
        try:
            df = pd.read_csv(file_infos[0]["datapath"])
        except Exception as e:
            return f"Error reading CSV: {e}"
            
        if 'unique_id' not in df.columns:
            return "Error: CSV must contain a 'unique_id' column."
            
        # Create Output Directories
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_folder = os.path.join(dest_base, f"Consolidated_Analysis_{timestamp}")
        img_out_folder = os.path.join(out_folder, "consolidated_images")
        
        try:
            os.makedirs(img_out_folder, exist_ok=True)
        except Exception as e:
            return f"Error creating output folders: {e}"
        
        # Add tracking columns if they don't exist
        for col in ['mean_temp_C', 'sd_temp_C', 'mean_temp_F', 'sd_temp_F', 'images_processed']:
            if col not in df.columns:
                df[col] = np.nan
        
        plots_processed = 0
        plots_missing = 0
        
        # Process each plot ID in the Field Book
        for idx, row in df.iterrows():
            uid = str(row['unique_id'])
            uid_esc = re.escape(uid)
            
            # Match pattern: <unique_id>_<YYYYMMDD_HHMMSS>_image_<number>.tiff
            pattern = re.compile(rf"^{uid_esc}_(\d{{8}}_\d{{6}})_image_\d+\.tiff$")
            matched_files = {} 
            
            # Scan the parent directory and all subdirectories automatically
            for root, _, files in os.walk(parent_dir):
                for file in files:
                    match = pattern.match(file)
                    if match:
                        ts = match.group(1) # Extract the timestamp grouping
                        full_path = os.path.join(root, file)
                        if ts not in matched_files:
                            matched_files[ts] = []
                        matched_files[ts].append(full_path)
                            
            if not matched_files:
                plots_missing += 1
                continue
                
            # Find the most recent timestamp capture burst 
            latest_ts = sorted(matched_files.keys(), reverse=True)[0]
            
            # Sort files alphanumerically
            latest_files = sorted(matched_files[latest_ts])
            
            # Enforce the target n_i constraint
            selected_files = latest_files[:n_i]
            
            c_means, c_sds, f_means, f_sds = [], [], [], []
            
            # Extract Thermal Stats from selected raw images
            for img_path in selected_files:
                filename = os.path.basename(img_path)
                dest_path = os.path.join(img_out_folder, filename)
                shutil.copy2(img_path, dest_path)
                
                img_data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img_data is not None:
                    raw_mean = np.mean(img_data)
                    raw_sd = np.std(img_data, ddof=1) if img_data.size > 1 else 0
                    
                    c_m = (raw_mean * 0.01) - 273.15
                    c_s = raw_sd * 0.01
                    f_m = (c_m * 1.8) + 32
                    f_s = c_s * 1.8
                    
                    c_means.append(c_m)
                    c_sds.append(c_s)
                    f_means.append(f_m)
                    f_sds.append(f_s)
            
            if c_means:
                df.at[idx, 'mean_temp_C'] = np.mean(c_means)
                df.at[idx, 'sd_temp_C'] = np.mean(c_sds)
                df.at[idx, 'mean_temp_F'] = np.mean(f_means)
                df.at[idx, 'sd_temp_F'] = np.mean(f_sds)
                df.at[idx, 'images_processed'] = len(c_means)
                plots_processed += 1
            else:
                plots_missing += 1

        out_csv_path = os.path.join(out_folder, "consolidated_thermal_data.csv")
        try:
            df.to_csv(out_csv_path, index=False)
        except Exception as e:
            return f"Error saving consolidated CSV: {e}"
        
        return (
            f"✅ Consolidation Complete!\n"
            f"----------------------------------------\n"
            f"Total plots in Field Book: {len(df)}\n"
            f"Plots successfully processed: {plots_processed}\n"
            f"Plots with missing images: {plots_missing}\n\n"
            f"Output Data saved to:\n"
            f"📄 {out_csv_path}\n"
            f"📁 {img_out_folder}"
        )

app = App(app_ui, server)