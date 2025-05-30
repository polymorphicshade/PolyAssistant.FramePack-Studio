import gradio as gr
import os
import sys
import torch
import devicetorch
import traceback
import gc

from torchvision.transforms.functional import rgb_to_grayscale
import types
functional_tensor_mod = types.ModuleType('functional_tensor')
functional_tensor_mod.rgb_to_grayscale = rgb_to_grayscale
sys.modules.setdefault('torchvision.transforms.functional_tensor', functional_tensor_mod)
    
from modules.toolbox.toolbox_processor import VideoProcessor
from modules.toolbox.message_manager import MessageManager
from modules.toolbox.system_monitor import SystemMonitor
from modules.settings import Settings

from diffusers_helper.memory import unload_complete_models, cpu, offload_model_from_device_for_memory_preservation

tb_message_mgr = MessageManager()
settings_instance = Settings()
tb_processor = VideoProcessor(tb_message_mgr, settings_instance) # Pass settings to VideoProcessor

def tb_update_messages():
    return tb_message_mgr.get_messages()

def tb_handle_update_monitor():
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

def tb_handle_extract_frames(video_path, extraction_rate, progress=gr.Progress()):
    tb_message_mgr.clear()
    tb_processor.tb_extract_frames(video_path, int(extraction_rate), progress)
    return tb_update_messages()

def tb_handle_reassemble_frames(uploaded_frames_dir_info, output_fps, progress=gr.Progress()):
    tb_message_mgr.clear()
    if uploaded_frames_dir_info is None or not uploaded_frames_dir_info:
        tb_message_mgr.add_warning("No frame directory selected or directory is empty/invalid.")
        return None, tb_update_messages()
    output_video = tb_processor.tb_reassemble_frames_to_video(uploaded_frames_dir_info, output_fps, progress)
    return output_video, tb_update_messages()

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

def tb_handle_unload_studio_models():
    tb_message_mgr.clear()
    print("Attempting to unload the main Transformer model...")
    tb_message_mgr.add_message("Attempting to unload the main Transformer model...")

    studio_scope = None
    try:
        if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'current_generator'):
            studio_scope = sys.modules['__main__']
        elif 'studio' in sys.modules and hasattr(sys.modules['studio'], 'current_generator'): # If studio was imported
             studio_scope = sys.modules['studio']
        else: # Last resort, try to import it if not already
            try:
                import studio as studio_module_direct_import
                if hasattr(studio_module_direct_import, 'current_generator'):
                    studio_scope = studio_module_direct_import
            except ImportError: pass
        
        if not studio_scope:
            tb_message_mgr.add_error("Unload Failed: Scope inaccessible.")
            return tb_update_messages()

        _cg = getattr(studio_scope, 'current_generator', None)
        if not (_cg and hasattr(_cg, 'transformer') and _cg.transformer):
            tb_message_mgr.add_message("No active Transformer found to unload.")
            print("No active transformer to unload.")
            return tb_update_messages()

        _trans_name = _cg.transformer.__class__.__name__ # Get name before it's gone
        _trans = _cg.transformer
        
        if hasattr(_cg, 'unload_loras'): _cg.unload_loras()
        
        if hasattr(_trans, 'to') and getattr(_trans, 'device', cpu) != cpu:
            try: _trans.to(cpu)
            except Exception as e_cpu: print(f"Transformer to CPU failed: {e_cpu}") # Log but continue

        _cg.transformer = None
        setattr(studio_scope, 'current_generator', None)

        del _trans
        del _cg
        gc.collect()
        devicetorch.empty_cache(torch) 

        print(f"Transformer '{_trans_name}' unload process completed.")
        tb_message_mgr.add_success(f"Success: Transformer '{_trans_name}' unloaded.")

    except Exception as e:
        print(f"Transformer Unload FAILED. Error:\n{traceback.format_exc()}")
        tb_message_mgr.add_error(f"Unload FAILED: Error. Check console. ({type(e).__name__})")
    
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
    
# --- Gradio Interface ---

def tb_create_video_toolbox_ui():
    
    # Determine initial autosave state.
    # You can make "toolbox_autosave_enabled" a persistent setting if desired.
    # For now, it defaults to True if not found in settings.
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
                    tb_manual_save_btn = gr.Button( # Assign to global for tb_handle_autosave_toggle
                        "💾 Save to Permanent Folder", 
                        variant="secondary", 
                        scale=3, 
                        visible=False # Autosave is off by default
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
                    tb_unload_studio_models_btn = gr.Button(
                        "📤 Unload Studio Models", variant="stop")
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
                        with gr.Column():
                            gr.Markdown("### Extract Frames from Video")
                            gr.Markdown("Extract frames from the **uploaded video (top-left)** as images.")
                            tb_extract_rate_slider = gr.Number(
                                label="Extract Every Nth Frame", 
                                value=1, 
                                minimum=1, 
                                step=1, 
                                info="1 = all frames. N = 1st, (N+1)th... (frame 0, N, 2N...)"
                            )
                            tb_extract_frames_btn = gr.Button("🔨 Extract Frames", variant="primary")
                        with gr.Column():
                            gr.Markdown("### Reassemble Frames to Video")
                            gr.Markdown("Drag individual frames or Click to upload a folder containing frame images to create a video.")
                            tb_reassemble_frames_input_files = gr.File( 
                                label="Click to Select Directory Containing Frame Images (e.g., PNG, JPG)", 
                                file_count="directory",
                                elem_classes="file-upload-area" 
                            )
                            tb_reassemble_output_fps = gr.Number(
                                label="Output Video FPS", value=25, minimum=1, step=1
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

        tb_extract_frames_btn.click(
            fn=tb_handle_extract_frames,
            inputs=[tb_input_video_component, tb_extract_rate_slider],
            outputs=[tb_message_output]
        )

        tb_reassemble_frames_btn.click(
            fn=tb_handle_reassemble_frames,
            inputs=[tb_reassemble_frames_input_files, tb_reassemble_output_fps], 
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
        tb_unload_studio_models_btn.click(
            fn=tb_handle_unload_studio_models,
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
    return tb_toolbox_ui_main_container, tb_input_video_component
