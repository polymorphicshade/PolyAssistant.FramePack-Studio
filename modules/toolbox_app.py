import gradio as gr
import os
import sys
import torch
import devicetorch
import traceback
import gc
import psutil

# patch fix for basicsr
from torchvision.transforms.functional import rgb_to_grayscale
import types
functional_tensor_mod = types.ModuleType('functional_tensor')
functional_tensor_mod.rgb_to_grayscale = rgb_to_grayscale
sys.modules.setdefault('torchvision.transforms.functional_tensor', functional_tensor_mod)
    
from modules.toolbox.toolbox_processor import VideoProcessor
from modules.toolbox.message_manager import MessageManager
from modules.toolbox.system_monitor import SystemMonitor
from modules.settings import Settings

try:
    from diffusers_helper.memory import cpu
except ImportError:
    print("WARNING: Could not import cpu from diffusers_helper.memory. Falling back to torch.device('cpu')")
    cpu = torch.device('cpu')


tb_message_mgr = MessageManager()
settings_instance = Settings()
tb_processor = VideoProcessor(tb_message_mgr, settings_instance) # Pass settings to VideoProcessor

def tb_update_messages():
    return tb_message_mgr.get_messages()

def tb_handle_update_monitor(): # This updates the TOOLBOX TAB's monitor
    return SystemMonitor.get_system_info()

def tb_handle_analyze_video(video_path):
    tb_message_mgr.clear()
    analysis = tb_processor.tb_analyze_video_input(video_path)
    return tb_update_messages(), analysis

def tb_handle_process_frames(video_path, fps_mode, speed_factor, progress=gr.Progress()):
    tb_message_mgr.clear()
    output_video = tb_processor.tb_process_frames(video_path, fps_mode, speed_factor, progress)
    return output_video, tb_update_messages()

def tb_handle_create_loop(video_path, loop_type, num_loops):
    tb_message_mgr.clear()
    output_video = tb_processor.tb_create_loop(video_path, loop_type, num_loops)
    return output_video, tb_update_messages()

def tb_handle_apply_filters(video_path, brightness, contrast, saturation, temperature,
                         sharpen, blur, denoise, vignette,
                         s_curve_contrast, film_grain_strength,
                         progress=gr.Progress()):
    tb_message_mgr.clear()
    output_video = tb_processor.tb_apply_filters(video_path, brightness, contrast, saturation, temperature, 
                                          sharpen, blur, denoise, vignette,
                                          s_curve_contrast, film_grain_strength, progress)
    return output_video, tb_update_messages()

# MODIFIED tb_handle_reassemble_frames
def tb_handle_reassemble_frames(
    selected_extracted_folder, # New input from dropdown
    uploaded_frames_dir_info,  # Existing input from gr.File
    output_fps,
    output_video_name, # New input from Textbox
    progress=gr.Progress()
):
    tb_message_mgr.clear()
    
    frames_source_to_use = None
    source_description = ""

    if selected_extracted_folder and selected_extracted_folder.strip():
        # Dropdown selection takes precedence
        frames_source_to_use = os.path.join(tb_processor.extracted_frames_target_path, selected_extracted_folder)
        source_description = f"selected folder '{selected_extracted_folder}'"
        if not os.path.isdir(frames_source_to_use):
            tb_message_mgr.add_error(f"Selected folder '{selected_extracted_folder}' not found at expected path: {frames_source_to_use}")
            return None, tb_update_messages()
    elif uploaded_frames_dir_info and (isinstance(uploaded_frames_dir_info, list) and uploaded_frames_dir_info):
        # Fallback to gr.File upload if dropdown is not used
        frames_source_to_use = uploaded_frames_dir_info
        source_description = "uploaded files/folder"
    else:
        tb_message_mgr.add_warning("No frame source selected or provided (neither dropdown nor file upload).")
        return None, tb_update_messages()

    tb_message_mgr.add_message(f"Attempting to reassemble frames from {source_description}.")
    output_video = tb_processor.tb_reassemble_frames_to_video(
        frames_source_to_use, 
        output_fps, 
        output_base_name_override=output_video_name, # Pass the desired name
        progress=progress
    )
    return output_video, tb_update_messages()

# NEW HANDLERS for extracted frames folder management
def tb_handle_extract_frames(video_path, extraction_rate, progress=gr.Progress()):
    tb_message_mgr.clear()
    tb_processor.tb_extract_frames(video_path, int(extraction_rate), progress)
    return tb_update_messages()
    
def tb_handle_refresh_extracted_folders():
    # tb_message_mgr.clear() # Optional: clear messages before refresh
    folders = tb_processor.tb_get_extracted_frame_folders()
    # Disable clear button if no folders, enable if folders exist and one might be selected
    clear_btn_update = gr.update(interactive=False) # Default to disabled
    if folders:
        # We don't know if one is selected yet, so keep it disabled until selection.
        # Or, if you want to enable it if *any* folder exists, change logic here.
        pass # Keep it disabled for now. It will be enabled on dropdown.change
    return gr.update(choices=folders, value=None), tb_update_messages(), clear_btn_update

def tb_handle_clear_selected_folder(selected_folder_to_delete):
    tb_message_mgr.clear()
    if not selected_folder_to_delete:
        tb_message_mgr.add_warning("No folder selected from the dropdown to delete.")
        return tb_update_messages(), gr.update() # Return current dropdown state

    success = tb_processor.tb_delete_extracted_frames_folder(selected_folder_to_delete)
    
    # After deletion, refresh the folder list
    updated_folders = tb_processor.tb_get_extracted_frame_folders()
    # If the deleted folder was the one selected, the value should clear.
    # The dropdown will clear itself if its current value is no longer in choices.
    return tb_update_messages(), gr.update(choices=updated_folders, value=None)
    
# --- Default Filter Values ---
TB_DEFAULT_FILTER_SETTINGS = { # Prefixed constant name
    "brightness": 0, "contrast": 1, "saturation": 1, "temperature": 0,
    "vignette": 0, "sharpen": 0, "blur": 0, "denoise": 0,
    "s_curve_contrast": 0, "film_grain_strength": 0
}

def tb_update_filter_sliders_from_preset(preset_name): # Prefixed internal helper
    preset_settings_map = {
        "none": TB_DEFAULT_FILTER_SETTINGS.copy(), 
        "cinematic": {"brightness": -5, "contrast": 1.3, "saturation": 0.9, "temperature": 20, "vignette": 10, "sharpen": 1.2, "blur": 0, "denoise": 0, "s_curve_contrast": 15, "film_grain_strength": 5}, 
        "vintage": {"brightness": 5, "contrast": 1.1, "saturation": 0.7, "temperature": 15, "vignette": 30, "sharpen": 0, "blur": 0.5, "denoise": 0, "s_curve_contrast": 10, "film_grain_strength": 10}, 
        "cool": {"brightness": 0, "contrast": 1.2, "saturation": 1.1, "temperature": -15, "vignette": 0, "sharpen": 1.0, "blur": 0, "denoise": 0, "s_curve_contrast": 5, "film_grain_strength": 0}, 
        "warm": {"brightness": 5, "contrast": 1.1, "saturation": 1.2, "temperature": 20, "vignette": 0, "sharpen": 0, "blur": 0, "denoise": 0, "s_curve_contrast": 5, "film_grain_strength": 0}, 
        "dramatic": {"brightness": -5, "contrast": 1.2, "saturation": 0.9, "temperature": -10, "vignette": 20, "sharpen": 1.2, "blur": 0, "denoise": 0, "s_curve_contrast": 20, "film_grain_strength": 8} 
    }
    selected_preset = preset_settings_map.get(preset_name, TB_DEFAULT_FILTER_SETTINGS.copy())
    settings = {**TB_DEFAULT_FILTER_SETTINGS, **selected_preset}

    return settings["brightness"], settings["contrast"], settings["saturation"], settings["temperature"], \
           settings["sharpen"], settings["blur"], settings["denoise"], settings["vignette"], \
           settings["s_curve_contrast"], settings["film_grain_strength"]

def tb_handle_reset_all_filters():
    tb_message_mgr.add_message("Filter sliders reset to default values.")
    return TB_DEFAULT_FILTER_SETTINGS["brightness"], TB_DEFAULT_FILTER_SETTINGS["contrast"], \
           TB_DEFAULT_FILTER_SETTINGS["saturation"], TB_DEFAULT_FILTER_SETTINGS["temperature"], \
           TB_DEFAULT_FILTER_SETTINGS["sharpen"], TB_DEFAULT_FILTER_SETTINGS["blur"], \
           TB_DEFAULT_FILTER_SETTINGS["denoise"], TB_DEFAULT_FILTER_SETTINGS["vignette"], \
           TB_DEFAULT_FILTER_SETTINGS["s_curve_contrast"], TB_DEFAULT_FILTER_SETTINGS["film_grain_strength"], \
           "none", tb_update_messages() 
           
def tb_handle_upscale_video(video_path, upscale_factor, progress=gr.Progress()):
    tb_message_mgr.clear()
    if video_path is None:
        tb_message_mgr.add_warning("No input video selected for upscaling.")
        return None, tb_update_messages()
    
    output_video = tb_processor.tb_upscale_video(video_path, upscale_factor, progress)
    return output_video, tb_update_messages()

def tb_handle_delete_studio_transformer():
    tb_message_mgr.clear()
    tb_message_mgr.add_message("Attempting to directly access and delete Studio transformer...")
    print("Attempting to directly access and delete Studio transformer...")
    log_messages_from_action = [] # This will store detailed steps for the UI log

    studio_module_instance = None
    # Check __main__ first
    if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'current_generator'):
        studio_module_instance = sys.modules['__main__']
        print("Found studio context in __main__.") # Console log only
    elif 'studio' in sys.modules and hasattr(sys.modules['studio'], 'current_generator'):
        studio_module_instance = sys.modules['studio']
        print("Found studio context in sys.modules['studio'].") # Console log only
    else:
        print("ERROR: Could not find the 'studio' module's active context.")
        # Add error messages directly for UI, as log_messages_from_action won't be normally processed
        tb_message_mgr.add_message("ERROR: Could not find the 'studio' module's active context in sys.modules.")
        tb_message_mgr.add_error("Deletion Failed: Studio module context not found.")
        return tb_update_messages()

    generator_object_to_delete = getattr(studio_module_instance, 'current_generator', None)
    print(f"Direct access: generator_object_to_delete is {type(generator_object_to_delete)}, id: {id(generator_object_to_delete)}")

    if generator_object_to_delete is not None:
        model_name_str = "Unknown Model"
        try:
            if hasattr(generator_object_to_delete, 'get_model_name') and callable(generator_object_to_delete.get_model_name):
                model_name_str = generator_object_to_delete.get_model_name()
            elif hasattr(generator_object_to_delete, 'transformer') and generator_object_to_delete.transformer is not None:
                model_name_str = generator_object_to_delete.transformer.__class__.__name__
            else:
                model_name_str = generator_object_to_delete.__class__.__name__
        except Exception:
            pass # model_name_str remains "Unknown Model"
        
        tb_message_mgr.add_message(f" Deletion of '{model_name_str}' initiated.")
        
        log_messages_from_action.append(f" Found active generator: {model_name_str}. Preparing for deletion.")
        print(f"Found active generator: {model_name_str}. Preparing for deletion.")

        try:
            # Step 1: Unload LoRAs
            if hasattr(generator_object_to_delete, 'unload_loras') and callable(generator_object_to_delete.unload_loras):
                print("   - LoRAs: Unloading from transformer...")
                generator_object_to_delete.unload_loras()
            else:
                log_messages_from_action.append("    - LoRAs: No unload method found or not applicable.")

            # Step 2: Operate on the transformer object
            if hasattr(generator_object_to_delete, 'transformer') and generator_object_to_delete.transformer is not None:
                transformer_object_ref = generator_object_to_delete.transformer
                transformer_name_for_log = transformer_object_ref.__class__.__name__ # Still useful for detailed prints/errors
                print(f"   - Transformer ({transformer_name_for_log}): Preparing for memory operations.")

                moved_to_cpu_successfully = False
                if hasattr(transformer_object_ref, 'device') and transformer_object_ref.device != cpu:
                    if hasattr(transformer_object_ref, 'to') and callable(transformer_object_ref.to):
                        try:
                            print(f"   - Transformer ({transformer_name_for_log}): Moving to CPU...")
                            transformer_object_ref.to(cpu)
                            # ℹ️    - Transformer moved to CPU.
                            log_messages_from_action.append("    - Transformer moved to CPU.")
                            print(f"   - Transformer ({transformer_name_for_log}): Moved to CPU.")
                            moved_to_cpu_successfully = True
                        except Exception as e_cpu:
                            error_msg_cpu = f"    - Transformer ({transformer_name_for_log}): Move to CPU FAILED: {e_cpu}"
                            log_messages_from_action.append(error_msg_cpu)
                            print(error_msg_cpu) # Keep detailed error for console
                    else:
                        log_messages_from_action.append(f"    - Transformer ({transformer_name_for_log}): Cannot move to CPU, 'to' method not found.")
                        print(f"   - Transformer ({transformer_name_for_log}): Cannot move to CPU, 'to' method not found.")
                elif hasattr(transformer_object_ref, 'device') and transformer_object_ref.device == cpu:
                     log_messages_from_action.append("    - Transformer already on CPU.")
                     print(f"   - Transformer ({transformer_name_for_log}): Already on CPU.")
                     moved_to_cpu_successfully = True # Considered successful for deletion purposes
                else: # No device attribute or other case
                    log_messages_from_action.append("    - Transformer: Could not determine device or move to CPU.")
                    print(f"   - Transformer ({transformer_name_for_log}): Could not determine device or move to CPU.")
                
                print(f"   - Transformer ({transformer_name_for_log}): Removing attribute from generator...")
                generator_object_to_delete.transformer = None
                print(f"   - Transformer ({transformer_name_for_log}): Deleting Python reference...")
                del transformer_object_ref
                # ℹ️    - Transformer reference deleted.
                log_messages_from_action.append("    - Transformer reference deleted.")
                print(f"   - Transformer ({transformer_name_for_log}): Reference deleted.")
            else:
                log_messages_from_action.append("    - Transformer: Not found or already unloaded.")
                print("   - Transformer: Not found or already unloaded.")

            # Step 3: Nullify the global reference in studio module
            generator_class_name_for_log = generator_object_to_delete.__class__.__name__
            print(f"   - Model Generator ({generator_class_name_for_log}): Setting global reference to None...")
            setattr(studio_module_instance, 'current_generator', None)
            log_messages_from_action.append("    - 'current_generator' in studio module set to None.")
            print("   - Global 'current_generator' in studio module successfully set to None.")

            # Step 4: Delete our Python reference to the generator object itself
            print(f"   - Model Generator ({generator_class_name_for_log}): Deleting local Python reference...")
            del generator_object_to_delete # Actual deletion
            print(f"   - Model Generator ({generator_class_name_for_log}): Python reference deleted.")

            # Step 5: System Cleanup
            print("   - System: Performing garbage collection and CUDA cache clearing.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_messages_from_action.append("    - GC and CUDA cache cleared.")
            print("   - System: GC and CUDA cache clear completed.")
            
            log_messages_from_action.append(f"✅ Deletion of '{model_name_str}' completed successfully from toolbox.")
            tb_message_mgr.add_success(f"Deletion of '{model_name_str}' initiated from toolbox.")

        except Exception as e_del:
            error_msg_del = f"Error during deletion process: {e_del}"
            log_messages_from_action.append(f"    - {error_msg_del}") # Add error to detailed log
            print(f"   - {error_msg_del}")
            traceback.print_exc()
            tb_message_mgr.add_error(f"Deletion Error: {e_del}")
    else:
        tb_message_mgr.add_message("ℹ️ No active generator found. Nothing to delete.") # Direct UI message
        print("No active generator found via direct access. Nothing to delete.")

    # Add all collected detailed log messages to the UI
    for msg_item in log_messages_from_action:
        tb_message_mgr.add_message(msg_item)

    return tb_update_messages()

def tb_handle_manually_save_video(temp_video_path_from_component):
    tb_message_mgr.clear()
    if not temp_video_path_from_component:
        tb_message_mgr.add_warning("No video in the output player to save.")
        # Return the current value of the component (which is None or empty)
        return temp_video_path_from_component, tb_update_messages()
    
    # tb_copy_video_to_permanent_storage now returns the path in the permanent store if successful,
    # or the original temp_video_path if copy failed or source was invalid.
    # The key is that the source temp_video_path_from_component is NOT deleted by this operation.
    copied_path = tb_processor.tb_copy_video_to_permanent_storage(temp_video_path_from_component)
    
    # We don't need to change the tb_processed_video_output component's value,
    # as the temp file is still there and playable.
    # Just update the messages.
    if copied_path and os.path.abspath(copied_path) != os.path.abspath(temp_video_path_from_component):
        # This means copy was successful and a new file exists at copied_path
        tb_message_mgr.add_success(f"Video successfully copied to permanent storage.") # tb_copy already adds detailed msg
    # If copy failed, tb_copy_video_to_permanent_storage would have added an error message.
    
    return temp_video_path_from_component, tb_update_messages() # Always return original temp path to keep video in player

def tb_handle_clear_temp_files():
    tb_message_mgr.clear() # Clear messages for this new operation
    success = tb_processor.tb_clear_temporary_files()
    
    # Regardless of success/failure of deletion, we clear the output video component
    # as its source file (if any) would be gone or the dir structure changed.
    if success:
        tb_message_mgr.add_success("Temporary files cleared.") # General success message
    else:
        tb_message_mgr.add_warning("Issue during temporary file cleanup. Check messages.")

    return None, tb_update_messages() # Clear tb_processed_video_output, update messages

# This function IS EXPORTED and used by interface.py
def tb_get_formatted_toolbar_stats():
    """Fetches and formats System RAM, NVIDIA VRAM, Temp, and Load for the main toolbar textboxes."""
    vram_full_str = "VRAM: N/A"
    gpu_full_str = "GPU: N/A"
    ram_full_str = "RAM: N/A" 
    
    vram_component_visible = False 
    gpu_component_visible = False

    try:
        ram_info_psutil = psutil.virtual_memory() 
        ram_used_gb = ram_info_psutil.used / (1024**3)
        ram_total_gb = ram_info_psutil.total / (1024**3)
        ram_full_str = f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB ({ram_info_psutil.percent}%)"

        if torch.cuda.is_available(): 
            _, nvidia_metrics, _ = SystemMonitor.get_nvidia_gpu_info()
            if nvidia_metrics:
                vram_used = nvidia_metrics.get('memory_used_gb', 0.0)
                vram_total = nvidia_metrics.get('memory_total_gb', 0.0)
                vram_full_str = f"VRAM: {vram_used:.1f}/{vram_total:.1f}GB"
                vram_component_visible = True
                
                temp = nvidia_metrics.get('temperature', 0.0)
                load = nvidia_metrics.get('utilization', 0.0)
                gpu_full_str = f"GPU: {temp:.0f}°C {load:.0f}%"
                gpu_component_visible = True
        
    except Exception as e:
        print(f"Error getting system stats values for toolbar (from toolbox_app.py): {e}")
        ram_full_str = "RAM: Error"
        is_nvidia_expected = torch.cuda.is_available()
        if is_nvidia_expected:
            vram_full_str = "VRAM: Error"
            gpu_full_str = "GPU: Error"
            vram_component_visible = True 
            gpu_component_visible = True
        else:
            vram_full_str = "VRAM: N/A" 
            gpu_full_str = "GPU: N/A"
            vram_component_visible = False
            gpu_component_visible = False
        
    # If a component is not visible, we can send an empty string or its "N/A" string,
    # as the `visible` flag in gr.update will handle hiding it.
    # For consistency, let's send the appropriate string even if it's about to be hidden.
    
    return (
        gr.update(value=ram_full_str), 
        gr.update(value=vram_full_str, visible=vram_component_visible),
        gr.update(value=gpu_full_str, visible=gpu_component_visible)
    )

   
# --- Gradio Interface ---

def tb_create_video_toolbox_ui():
    initial_autosave_state = settings_instance.get("toolbox_autosave_enabled", True)
    tb_processor.set_autosave_mode(initial_autosave_state)
    
    with gr.Column() as tb_toolbox_ui_main_container:
        with gr.Row():
            with gr.Column(scale=1):
                tb_input_video_component = gr.Video(
                    label="Upload Video for post-processing",
                    elem_classes="video-size",
                    elem_id="toolbox-video-player"                    
                )
                tb_analyze_button = gr.Button("📊 Analyze Video")

            with gr.Column(scale=1):
                tb_processed_video_output = gr.Video(
                    label="Processed Video",
                    interactive=False,
                    elem_classes="video-size" 
                )
                with gr.Row():
                    tb_use_processed_as_input_btn = gr.Button("🔄 Use Processed as Input", scale=3)
                    tb_manual_save_btn = gr.Button( 
                        "💾 Save to Permanent Folder", 
                        variant="secondary", 
                        scale=3, 
                        visible=not initial_autosave_state
                    )
                    tb_autosave_checkbox = gr.Checkbox(
                        label="Autosave", 
                        value=initial_autosave_state,
                        scale=1
                    )
        with gr.Row():
            with gr.Column(scale=1):
                tb_video_analysis_output = gr.Textbox(
                    label="Video Analysis",
                    lines=12,
                    interactive=False,
                    elem_classes="analysis-box", 
                )
            with gr.Column(scale=1):
                tb_resource_monitor_output = gr.Textbox(
                    label="💻 System Monitor",
                    lines=9,
                    interactive=False,
                )
                with gr.Row():
                    tb_delete_studio_transformer_btn = gr.Button(
                        "📤 Unload Studio Model", variant="stop")
                    gr.Markdown(
                        "Studio will automatically reload models when you start a new video generation. "
                    )
        with gr.Accordion("Operations", open=True):
            with gr.Tabs():
                with gr.TabItem("🎞️ Frame Adjust (Speed & Interpolation)"):
                    gr.Markdown("Adjust video speed and interpolate frames using RIFE AI.")
                    tb_process_fps_mode = gr.Radio(
                        choices=["No Interpolation", "2x RIFE Interpolation"],
                        value="No Interpolation",
                        label="RIFE Frame Interpolation",
                        info="Select '2x RIFE Interpolation' to double the frame rate."
                    )
                    tb_process_speed_factor = gr.Slider(
                        minimum=0.25, maximum=4.0, step=0.05, value=1.0, label="Adjust Video Speed Factor"
                    )
                    tb_process_frames_btn = gr.Button("🚀 Process Frames", variant="primary")

                with gr.TabItem("🔄 Video Loop"):
                    gr.Markdown("Create looped or ping-pong versions of the video.")
                    tb_loop_type_select = gr.Radio(
                        choices=["loop", "ping-pong"], value="loop", label="Loop Type"
                    )
                    tb_num_loops_slider = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Number of Loops/Repeats")
                    tb_create_loop_btn = gr.Button("🔁 Create Loop", variant="primary")

                with gr.TabItem("🎨 Video Filters (FFmpeg)"):
                    gr.Markdown("Apply visual enhancements using FFmpeg filters (WIP).")
                    with gr.Row():
                        tb_filter_preset_select = gr.Radio(
                            choices=["none", "cinematic", "vintage", "cool", "warm", "dramatic"], value="none",
                            label="Style Presets", scale=3
                    )
                        tb_reset_filters_btn = gr.Button("🔄 Reset All Filters", scale=1)

                    with gr.Row():
                        tb_filter_brightness = gr.Slider(-100, 100, value=TB_DEFAULT_FILTER_SETTINGS["brightness"], step=1, label="Brightness (%)")
                        tb_filter_contrast = gr.Slider(0, 3, value=TB_DEFAULT_FILTER_SETTINGS["contrast"], step=0.05, label="Contrast (Linear)") 
                    with gr.Row():
                        tb_filter_saturation = gr.Slider(0, 3, value=TB_DEFAULT_FILTER_SETTINGS["saturation"], step=0.05, label="Saturation")
                        tb_filter_temperature = gr.Slider(-100, 100, value=TB_DEFAULT_FILTER_SETTINGS["temperature"], step=1, label="Color Temperature Adjust")
                    with gr.Row():
                        tb_filter_sharpen = gr.Slider(0, 5, value=TB_DEFAULT_FILTER_SETTINGS["sharpen"], step=0.1, label="Sharpen Strength")
                        tb_filter_blur = gr.Slider(0, 5, value=TB_DEFAULT_FILTER_SETTINGS["blur"], step=0.1, label="Blur Strength")
                    with gr.Row():
                        tb_filter_denoise = gr.Slider(0, 10, value=TB_DEFAULT_FILTER_SETTINGS["denoise"], step=0.1, label="Denoise Strength")
                        tb_filter_vignette = gr.Slider(0, 100, value=TB_DEFAULT_FILTER_SETTINGS["vignette"], step=1, label="Vignette Strength (%)")
                    with gr.Row():
                        tb_filter_s_curve_contrast = gr.Slider(0, 100, value=TB_DEFAULT_FILTER_SETTINGS["s_curve_contrast"], step=1, label="S-Curve Contrast")
                        tb_filter_film_grain_strength = gr.Slider(0, 50, value=TB_DEFAULT_FILTER_SETTINGS["film_grain_strength"], step=1, label="Film Grain Strength") 

                    tb_apply_filters_btn = gr.Button("✨ Apply Filters", variant="primary")
                
                with gr.TabItem("🖼️ Frames I/O"): 
                    with gr.Row():
                        with gr.Column(): # Column for extraction
                            gr.Markdown("### Extract Frames from Video")
                            gr.Markdown("Extract frames from the **uploaded video (top-left)** as images.")
                            tb_extract_rate_slider = gr.Number(
                                label="Extract Every Nth Frame", value=1, minimum=1, step=1, 
                                info="1 = all frames. N = 1st, (N+1)th... (frame 0, N, 2N...)"
                            )
                            tb_extract_frames_btn = gr.Button("🔨 Extract Frames", variant="primary")
                        
                        with gr.Column(): # Column for reassembly - UI as you provided
                            gr.Markdown("### Reassemble Frames to Video")
                            
                            tb_extracted_folders_dropdown = gr.Dropdown(
                                label="Select Previously Extracted Folder",
                                info="Select a folder from your 'extracted_frames' directory."
                            )
                            with gr.Row():
                                tb_refresh_extracted_folders_btn = gr.Button("🔄 Refresh List")
                                tb_clear_selected_folder_btn = gr.Button(
                                    "🗑️ Clear Selected Folder", variant="stop", interactive=False
                                )

                            gr.Markdown("Alternatively, drag individual frames or Click to upload a folder containing frame images:") # Label for this component
                            tb_reassemble_frames_input_files = gr.File( 
                                label="Upload Frame Images Folder (or individual image files)", # Updated label for clarity
                                file_count="directory", 
                            )
                            tb_reassemble_output_fps = gr.Number(
                                label="Output Video FPS", value=30, minimum=1, step=1
                            )
                            tb_reassemble_video_name_input = gr.Textbox( # NEW Textbox for name
                                label="Output Video Name (optional, .mp4 added automatically)"
                            )
                            tb_reassemble_frames_btn = gr.Button("🧩 Reassemble Video", variant="primary")
                            
                with gr.TabItem("📈 Upscale Video (ESRGAN)"):
                    gr.Markdown("Upscale video resolution using Real-ESRGAN.")
                    tb_upscale_factor_radio = gr.Radio(
                        choices=["2x", "4x"], 
                        value="2x",
                        label="Upscale Factor",
                        info="Select the desired upscaling factor."
                    )
                    tb_upscale_video_btn = gr.Button("🚀 Upscale Video", variant="primary")            
                    
        with gr.Row():
            tb_message_output = gr.Textbox(label="Console Messages", lines=10, interactive=False, elem_classes="message-box", value=tb_update_messages)
        with gr.Row():       
            tb_open_folder_button = gr.Button("📁 Open Output Folder", scale=4)
            tb_clear_temp_button = gr.Button("🗑️ Clear Temporary Files", variant="stop", scale=1)

        # --- Event Handlers ---
        tb_input_video_component.upload(fn=lambda: (tb_message_mgr.clear() or tb_update_messages(), None), outputs=[tb_message_output, tb_video_analysis_output])
        tb_input_video_component.clear(fn=lambda: (tb_message_mgr.clear() or tb_update_messages(), None, None), outputs=[tb_message_output, tb_video_analysis_output, tb_processed_video_output])

        tb_analyze_button.click(
            fn=tb_handle_analyze_video,
            inputs=[tb_input_video_component],
            outputs=[tb_message_output, tb_video_analysis_output]
        )

        tb_process_frames_btn.click(
            fn=tb_handle_process_frames,
            inputs=[tb_input_video_component, tb_process_fps_mode, tb_process_speed_factor],
            outputs=[tb_processed_video_output, tb_message_output]
        )

        tb_create_loop_btn.click(
            fn=tb_handle_create_loop,
            inputs=[tb_input_video_component, tb_loop_type_select, tb_num_loops_slider],
            outputs=[tb_processed_video_output, tb_message_output]
        )
        
        tb_all_filter_sliders = [ # Prefixed component names in list
            tb_filter_brightness, tb_filter_contrast, tb_filter_saturation, tb_filter_temperature,
            tb_filter_sharpen, tb_filter_blur, tb_filter_denoise, tb_filter_vignette,
            tb_filter_s_curve_contrast, tb_filter_film_grain_strength
        ]

        tb_filter_preset_select.change(
            fn=tb_update_filter_sliders_from_preset,
            inputs=[tb_filter_preset_select],
            outputs=tb_all_filter_sliders 
        )

        tb_reset_filters_btn.click(
            fn=tb_handle_reset_all_filters,
            inputs=None, 
            outputs=tb_all_filter_sliders + [tb_filter_preset_select, tb_message_output] 
        )

        tb_apply_filters_btn.click(
            fn=tb_handle_apply_filters,
            inputs=[tb_input_video_component] + tb_all_filter_sliders, 
            outputs=[tb_processed_video_output, tb_message_output]
        )

        tb_use_processed_as_input_btn.click(
            fn=lambda video: (video, tb_message_mgr.add_message("Moved processed video to input.") or tb_update_messages()),
            inputs=[tb_processed_video_output],
            outputs=[tb_input_video_component, tb_message_output]
        ).then(
            fn=lambda: (None, None), 
            outputs=[tb_processed_video_output, tb_video_analysis_output]
        )
        
        tb_upscale_video_btn.click(
            fn=tb_handle_upscale_video,
            inputs=[tb_input_video_component, tb_upscale_factor_radio],
            outputs=[tb_processed_video_output, tb_message_output]
        )
        
        # Frames I/O Event Handlers
        tb_extract_frames_btn.click(
            fn=tb_handle_extract_frames,
            inputs=[tb_input_video_component, tb_extract_rate_slider],
            outputs=[tb_message_output]
        ).then( # Automatically refresh folder list after extraction
            fn=tb_handle_refresh_extracted_folders,
            inputs=None,
            outputs=[tb_extracted_folders_dropdown, tb_message_output, tb_clear_selected_folder_btn]
        )

        tb_refresh_extracted_folders_btn.click(
            fn=tb_handle_refresh_extracted_folders,
            inputs=None,
            outputs=[tb_extracted_folders_dropdown, tb_message_output, tb_clear_selected_folder_btn]
        )

        tb_extracted_folders_dropdown.change(
            fn=lambda selection: gr.update(interactive=bool(selection)), # Enable/disable clear button
            inputs=[tb_extracted_folders_dropdown],
            outputs=[tb_clear_selected_folder_btn]
        )

        tb_clear_selected_folder_btn.click(
            fn=tb_handle_clear_selected_folder,
            inputs=[tb_extracted_folders_dropdown],
            outputs=[tb_message_output, tb_extracted_folders_dropdown] # Dropdown updates from handler
        ).then( # After clearing, re-evaluate clear button interactivity
            fn=lambda selection: gr.update(interactive=bool(selection)),
            inputs=[tb_extracted_folders_dropdown],
            outputs=[tb_clear_selected_folder_btn]
        )

        tb_reassemble_frames_btn.click(
            fn=tb_handle_reassemble_frames,
            inputs=[
                tb_extracted_folders_dropdown, 
                tb_reassemble_frames_input_files, 
                tb_reassemble_output_fps,
                tb_reassemble_video_name_input # New input for name
            ], 
            outputs=[tb_processed_video_output, tb_message_output] 
        )
        
        # Clear messages on new uploads to gr.File
        tb_reassemble_frames_input_files.upload(fn=lambda: tb_message_mgr.clear() or tb_update_messages(), outputs=tb_message_output)
        tb_reassemble_frames_input_files.clear(fn=lambda: tb_message_mgr.clear() or tb_update_messages(), outputs=tb_message_output)
        tb_open_folder_button.click(
            fn=lambda: tb_processor.tb_open_output_folder() or tb_update_messages(),
            outputs=[tb_message_output]
        )
        
        tb_monitor_timer = gr.Timer(2, active=True) 
        
        tb_monitor_timer.tick(
            fn=tb_handle_update_monitor,
            outputs=[tb_resource_monitor_output],
        )
        # ADDED: Event handler for the new unload button
        tb_delete_studio_transformer_btn.click(
            fn=tb_handle_delete_studio_transformer,
            inputs=[], # No Gradio inputs for this action
            outputs=[tb_message_output] # Update the toolbox's message console
        )
        # New event handler for the manual save button
        tb_manual_save_btn.click(
            fn=tb_handle_manually_save_video,
            inputs=[tb_processed_video_output],
            outputs=[tb_processed_video_output, tb_message_output] 
        ) 
        # Handler for the autosave checkbox
        def tb_handle_autosave_toggle(autosave_is_on_ui_value):
            tb_processor.set_autosave_mode(autosave_is_on_ui_value) # Update the processor's internal switch
            return {
                tb_manual_save_btn: gr.update(visible=not autosave_is_on_ui_value),
                tb_message_output: gr.update(value=tb_update_messages()) # Fetch new messages
            }

        tb_autosave_checkbox.change(
            fn=tb_handle_autosave_toggle,
            inputs=[tb_autosave_checkbox],
            # Outputs list refined to only what the handler updates
            outputs=[tb_manual_save_btn, tb_message_output]
        )
        # Event handler for the new clear temporary files button
        tb_clear_temp_button.click(
            fn=tb_handle_clear_temp_files,
            inputs=None, # No inputs needed
            outputs=[tb_processed_video_output, tb_message_output] # Clear video output, update messages
        )
    # MODIFIED RETURN SIGNATURE: Only return what's needed by interface.py
    # The toolbar Markdown components are no longer created or returned here.
    return tb_toolbox_ui_main_container, tb_input_video_component 
