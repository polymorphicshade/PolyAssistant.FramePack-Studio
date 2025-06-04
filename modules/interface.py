import gradio as gr
import time
import datetime
import random
import json
import os
import shutil
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import base64
import io

from modules.version import APP_VERSION, APP_VERSION_DISPLAY

import subprocess
import itertools
import re
from collections import defaultdict
import imageio
import imageio.plugins.ffmpeg
import ffmpeg
from diffusers_helper.utils import generate_timestamp

from modules.video_queue import JobStatus, Job
from modules.prompt_handler import get_section_boundaries, get_quick_prompts, parse_timestamped_prompt
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.bucket_tools import find_nearest_bucket

from modules.toolbox_app import tb_create_video_toolbox_ui, tb_get_formatted_toolbar_stats

def create_interface(
    process_fn,
    monitor_fn,
    end_process_fn,
    update_queue_status_fn,
    load_lora_file_fn,
    job_queue,
    settings,
    default_prompt: str = '[1s: The person waves hello] [3s: The person jumps up and down] [5s: The person does a dance]',
    lora_names: list = [],
    lora_values: list = []
):
    """
    Create the Gradio interface for the video generation application

    Args:
        process_fn: Function to process a new job
        monitor_fn: Function to monitor an existing job
        end_process_fn: Function to cancel the current job
        update_queue_status_fn: Function to update the queue status display
        default_prompt: Default prompt text
        lora_names: List of loaded LoRA names

    Returns:
        Gradio Blocks interface
    """
    # Get section boundaries and quick prompts
    section_boundaries = get_section_boundaries()
    quick_prompts = get_quick_prompts()

    #XY helper
    def generate_tab_xy_plot_process():
        return xy_plot_process_wrapper(settings, xy_plot_process,
            model_type.value, input_image.value, end_frame_image_original.value,
            end_frame_strength_original.value, latent_type.value,
            prompt.value, blend_sections.value, steps.value, total_second_length.value,
            resolutionW.value, resolutionH.value, seed.value, randomize_seed.value, use_teacache.value,
            teacache_num_steps.value, teacache_rel_l1_thresh.value, latent_window_size.value,
            cfg.value, gs.value, rs.value,
            xy_plot_axis_x_switch.value, xy_plot_axis_x_value_text.value, xy_plot_axis_x_value_dropdown.value,
            xy_plot_axis_y_switch.value, xy_plot_axis_y_value_text.value, xy_plot_axis_y_value_dropdown.value,
            xy_plot_axis_z_switch.value, xy_plot_axis_z_value_text.value, xy_plot_axis_z_value_dropdown.value
        )
    # Create the interface
    css = make_progress_bar_css()
    css += """
    /* Image container styling - more aggressive approach */
    .contain-image, .contain-image > div, .contain-image > div > img {
        object-fit: contain !important;
    }

    #non-mirrored-video {
        transform: scaleX(-1) !important;
    }
    
    /* Target all images in the contain-image class and its children */
    .contain-image img,
    .contain-image > div > img,
    .contain-image * img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
        max-height: 100% !important;
        max-width: 100% !important;
    }
    
    /* Additional selectors to override Gradio defaults */
    .gradio-container img,
    .gradio-container .svelte-1b5oq5x,
    .gradio-container [data-testid="image"] img {
        object-fit: contain !important;
    }
    
    /* Toolbar styling */
    #fixed-toolbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        z-index: 1000;
        background: #333;
        color: #fff;
        padding: 5px 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Responsive toolbar title */
    .toolbar-title {
        font-size: 1.4rem;
        margin: 0;
        color: white;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Toolbar Patreon link */
    .toolbar-patreon {
        margin: 0 0 0 20px;
        color: white;
        font-size: 0.9rem;
        white-space: nowrap;
        display: inline-block;
    }
    .toolbar-patreon a {
        color: white;
        text-decoration: none;
    }
    .toolbar-patreon a:hover {
        text-decoration: underline;
    }
    
    /* Responsive design for screens */
    @media (max-width: 1024px) {
        .toolbar-patreon {
            display: none;
        }
        .footer-patreon {
            display: block;
        }
    }
    
    @media (min-width: 1025px) {
        .footer-patreon {
            display: none;
        }
    }
    
    @media (max-width: 768px) {
        .toolbar-title {
            font-size: 1.1rem;
            max-width: 150px;
        }
        #fixed-toolbar {
            padding: 3px 6px;
            gap: 4px;
        }
        .toolbar-text {
            font-size: 0.75rem;
        }
    }
    
    @media (max-width: 480px) {
        .toolbar-title {
            font-size: 1rem;
            max-width: 120px;
        }
        #fixed-toolbar {
            padding: 2px 4px;
            gap: 2px;
        }
        .toolbar-text {
            font-size: 0.7rem;
        }
    }
    
    /* Button styling */
    #toolbar-add-to-queue-btn button {
        font-size: 14px !important;
        padding: 4px 16px !important;
        height: 32px !important;
        min-width: 80px !important;
    }
    .narrow-button {
        min-width: 40px !important;
        width: 40px !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .gr-button-primary {
        color: white;
    }
    
    /* Layout adjustments */
    body, .gradio-container {
        padding-top: 36px !important;
    }
    
    @media (max-width: 768px) {
        body, .gradio-container {
            padding-top: 32px !important;
        }
    }
    
    @media (max-width: 480px) {
        body, .gradio-container {
            padding-top: 28px !important;
        }
    }
    
    /* control sizing for tb_input_video_component */    
    .video-size video {
        max-height: 60vh;
        min-height: 300px !important;
        object-fit: contain;
    }

    /* hide the gr.Video source selection bar for tb_input_video_component */
    #toolbox-video-player .source-selection {
        display: none !important;
    }
    
    .analysis-box {
    }

    .message-box {
    } 

    /* Styling for Textboxes used as stat displays in the toolbar */
    .toolbar-stat-textbox {
        /* Reset Gradio Textbox defaults */
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0px 4px !important; /* Minimal padding */
        min-height: unset !important; /* Allow it to be as short as text */
        height: auto !important;
    }

    .toolbar-stat-textbox textarea { /* Target the actual textarea element */
        color: white !important;
        font-family: inherit !important; /* Inherit from parent */
        font-size: 0.75rem !important; /* Match .toolbar-text or desired */
        line-height: 1.2 !important;   /* Adjust for vertical alignment */
        padding: 0px !important;      /* Remove internal padding of textarea */
        white-space: nowrap !important;
        overflow: hidden !important; /* Hide scrollbars if any */
        resize: none !important; /* Disable textarea resizing */
        text-align: center !important;  /* Ensure text centered*/
        min-height: unset !important;
        height: auto !important;
        background-color: transparent !important;
    }
    """

    # Get the theme from settings
    current_theme = settings.get("gradio_theme", "default") # Use default if not found
    block = gr.Blocks(css=css, title="FramePack Studio", theme=current_theme).queue()

    with block:
        with gr.Row(elem_id="fixed-toolbar"):
            with gr.Column(scale=0, min_width=400): # Title/Patreon
                gr.HTML("""
                <div style="display: flex; align-items: center;">
                    <h1 class='toolbar-title'>FramePack Studio</h1>
                    <p class='toolbar-patreon'><a href='https://patreon.com/Colinu' target='_blank'>Support on Patreon</a></p>
                </div>
                """)
            with gr.Column(scale=0, min_width=40):
                refresh_stats_btn = gr.Button("⟳", elem_id="refresh-stats-btn", elem_classes="narrow-button")  
            with gr.Column(scale=1, min_width=180): # Queue Stats
                queue_stats_display = gr.Markdown("<p style='margin:0;color:white;' class='toolbar-text'>Queue: 0 | Running: 0 | Completed: 0</p>")
                
            # --- System Stats Display - Single gr.Textbox per stat ---
            with gr.Column(scale=0, min_width=205): # RAM Column
                toolbar_ram_display_component = gr.Textbox(
                    value="RAM: N/A", 
                    interactive=False, 
                    lines=1, 
                    max_lines=1,
                    show_label=False,
                    elem_classes="toolbar-stat-textbox"
                )
            with gr.Column(scale=0, min_width=160): # VRAM Column
                toolbar_vram_display_component = gr.Textbox(
                    value="VRAM: N/A", 
                    interactive=False, 
                    lines=1, 
                    max_lines=1,
                    show_label=False,
                    elem_classes="toolbar-stat-textbox"
                    # Visibility controlled by tb_get_formatted_toolbar_stats
                )
            with gr.Column(scale=0, min_width=140): # GPU Column
                toolbar_gpu_display_component = gr.Textbox(
                    value="GPU: N/A", 
                    interactive=False, 
                    lines=1, 
                    max_lines=1,
                    show_label=False,                     
                    elem_classes="toolbar-stat-textbox"
                    # Visibility controlled by tb_get_formatted_toolbar_stats
                )
            # --- End of System Stats Display ---
            
            with gr.Column(scale=0, min_width=50):
                version_display = gr.Markdown(f"<p style='margin:0;color:white;' class='toolbar-text'>{APP_VERSION_DISPLAY}</p>")   
            # --- End of Toolbar ---
            
        # Essential to capture main_tabs_component for later use by send_to_toolbox_btn
        with gr.Tabs(elem_id="main_tabs") as main_tabs_component:
            with gr.Tab("Generate", id="generate_tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_type = gr.Radio(
                            choices=["Original", "Original with Endframe", "F1", "Video", "Video with Endframe", "Video F1", "XY Plot"],
                            value="Original",
                            label="Generation Type"
                        )
                        with gr.Group(visible=False) as xy_group:  # Default visibility: True
                            xy_plot_axis_options = {
                                # "type": [
                                #     "dropdown(checkboxGroup), textbox or number", 
                                #     "empty if textbox, dtype if number, [] if dropdown", 
                                #     "standard values", 
                                #     "True if multi axis - like prompt replace, False is only on one axis - like steps"
                                # ],
                                "Nothing": ["nothing", "", "", True],
                                "Model type": ["dropdown", ["Original", "F1"], ["Original", "F1"], False],
                                "End frame influence": ["number", "float", "0.05-0.95[3]", False],
                                "Latent type": ["dropdown", ["Black", "White", "Noise", "Green Screen"], ["Black", "Noise"], False],
                                "Prompt add": ["textbox", "", "", True],
                                "Prompt replace": ["textbox", "", "", True],
                                "Blend sections": ["number", "int", "3-7 [3]", False],
                                "Steps": ["number", "int", "15-30 [3]", False],
                                "Seed": ["number", "int", "1000-10000 [3]", False],
                                "Use teacache": ["dropdown", [True, False], [True, False], False],
                                "TeaCache steps": ["number", "int", "5-25 [3]", False],
                                "TeaCache rel_l1_thresh": ["number", "float", "0.01-0.3 [3]", False],
                                # "CFG": ["number", "float", "", False],
                                "Distilled CFG Scale": ["number", "float", "5-15 [3]", False],
                                # "RS": ["number", "float", "", False],
                                # "Use weighted embeddings": ["dropdown", [True, False], [True, False], False],
                            }
                            text_to_base_keys = {
                                "Model type": "model_type",
                                "End frame influence": "end_frame_strength_original",
                                "Latent type": "latent_type",
                                "Prompt add": "prompt",
                                "Prompt replace": "prompt",
                                "Blend sections": "blend_sections",
                                "Steps": "steps",
                                "Seed": "seed",
                                "Use teacache": "use_teacache",
                                "TeaCache steps":"teacache_num_steps",
                                "TeaCache rel_l1_thresh":"teacache_rel_l1_thresh",
                                "Latent window size": "latent_window_size",
                                # "CFG": "",
                                "Distilled CFG Scale": "gs",
                                # "RS": "",
                                # "Use weighted embeddings": "",
                            }

                            def xy_plot_parse_input(text):
                                text = text.strip()
                                if ',' in text:
                                    return [x.strip() for x in text.split(",")]
                                match = re.match(r'^\s*(-?\d*\.?\d*)\s*-\s*(-?\d*\.?\d*)\s*\[\s*(\d+)\s*\]$', text)
                                if match:
                                    start, end, count = map(float, match.groups())
                                    result = np.linspace(start, end, int(count))
                                    if np.allclose(result, np.round(result)):
                                        result = np.round(result).astype(int)
                                    return result.tolist()
                                return []
                            def xy_plot_convert_loras_text(arrayT):
                                lora_pattern = r"<lora:([^:>]+):([-+]?\d*\.?\d+)>"
                                matches = re.findall(lora_pattern, arrayT["prompt"])
                                arrayT["prompt"] = re.sub(lora_pattern, '', arrayT["prompt"]).strip()
                                usedLoras = []
                                weightLoras = [1 for _ in range(len(arrayT["lora_loaded_names"]))]
                                for n, w in matches:
                                    if n in arrayT["lora_loaded_names"] and n not in usedLoras:
                                        usedLoras.append(n)
                                        weightLoras[arrayT["lora_loaded_names"].index(n)] = float(w)
                                    # print(n, w, v["lora_loaded_names"], arrayT["selected_loras"])
                                arrayT["selected_loras"] = usedLoras
                                arrayT["lora_values"] = weightLoras
                                return arrayT
                            def xy_plot_axis_change(updated_value_type):
                                if xy_plot_axis_options[updated_value_type][0] == "textbox" or xy_plot_axis_options[updated_value_type][0] == "number":
                                    return gr.update(visible=True, value=xy_plot_axis_options[updated_value_type][2]), gr.update(visible=False, value=[], choices=[])
                                elif xy_plot_axis_options[updated_value_type][0] == "dropdown":
                                    return gr.update(visible=False), gr.update(visible=True, value=xy_plot_axis_options[updated_value_type][2], choices=xy_plot_axis_options[updated_value_type][1])
                                else:
                                    return gr.update(visible=False), gr.update(visible=False, value=[], choices=[])
                            def xy_plot_process(
                                    model_type, input_image, end_frame_image_original, 
                                    end_frame_strength_original, latent_type, 
                                    prompt, blend_sections, steps, total_second_length, 
                                    resolutionW, resolutionH, seed, randomize_seed, use_teacache, 
                                    teacache_num_steps, teacache_rel_l1_thresh, latent_window_size, 
                                    cfg, gs, rs, gpu_memory_preservation, mp4_crf, 
                                    axis_x_switch, axis_x_value_text, axis_x_value_dropdown, 
                                    axis_y_switch, axis_y_value_text, axis_y_value_dropdown, 
                                    axis_z_switch, axis_z_value_text, axis_z_value_dropdown
                                    ):
                                # print(model_type, input_image, latent_type, 
                                #     prompt, blend_sections, steps, total_second_length, 
                                #     resolutionW, resolutionH, seed, randomize_seed, use_teacache, 
                                #     latent_window_size, cfg, gs, rs, gpu_memory_preservation, 
                                #     mp4_crf, 
                                #     axis_x_switch, axis_x_value_text, axis_x_value_dropdown, 
                                #     axis_y_switch, axis_y_value_text, axis_y_value_dropdown, 
                                #     axis_z_switch, axis_z_value_text, axis_z_value_dropdown, sep=", ")
                                if axis_x_switch == "Nothing" and axis_y_switch == "Nothing" and axis_z_switch == "Nothing":
                                    return "Not selected any axis for plot", gr.update()
                                if (axis_x_switch == "Nothing" or axis_y_switch == "Nothing") and axis_z_switch != "Nothing":
                                    return "For using Z axis, first use X and Y axis", gr.update()
                                if axis_x_switch == "Nothing" and axis_y_switch != "Nothing":
                                    return "For using Y axis, first use X axis", gr.update()
                                if xy_plot_axis_options[axis_x_switch][0] == "dropdown" and len(axis_x_value_dropdown) < 1:
                                    return "No values for axis X", gr.update()
                                if xy_plot_axis_options[axis_y_switch][0] == "dropdown" and len(axis_y_value_dropdown) < 1:
                                    return "No values for axis Y", gr.update()
                                if xy_plot_axis_options[axis_z_switch][0] == "dropdown" and len(axis_z_value_dropdown) < 1:
                                    return "No values for axis Z", gr.update()
                                if not xy_plot_axis_options[axis_x_switch][3]:
                                    if axis_x_switch == axis_y_switch: 
                                        return "Axis type on X and Y axis are same, you can't do that generation.<br>Multi axis supported only for \"Prompt add\" and \"Prompt replace\".", gr.update()
                                    if axis_x_switch == axis_z_switch: 
                                        return "Axis type on X and Z axis are same, you can't do that generation.<br>Multi axis supported only for \"Prompt add\" and \"Prompt replace\".", gr.update()
                                if not xy_plot_axis_options[axis_y_switch][3]:
                                    if axis_y_switch == axis_z_switch: 
                                        return "Axis type on Y and Z axis are same, you can't do that generation.<br>Multi axis supported only for \"Prompt add\" and \"Prompt replace\".", gr.update()

                                base_generator_vars = {
                                    "model_type": model_type,
                                    "input_image": input_image,
                                    "input_video": None,
                                    "end_frame_image_original": end_frame_image_original,
                                    "end_frame_strength_original": end_frame_strength_original,
                                    "prompt": prompt,
                                    "n_prompt": "",
                                    "seed": seed,
                                    "total_second_length": total_second_length,
                                    "latent_window_size": latent_window_size,
                                    "steps": steps,
                                    "cfg": cfg,
                                    "gs": gs,
                                    "rs": rs,
                                    "gpu_memory_preservation": gpu_memory_preservation,
                                    "use_teacache": use_teacache,
                                    "teacache_num_steps": teacache_num_steps,
                                    "teacache_rel_l1_thresh": teacache_rel_l1_thresh,
                                    "mp4_crf": mp4_crf,
                                    "randomize_seed_checked": False,
                                    "save_metadata_checked": True,
                                    "blend_sections": blend_sections,
                                    "latent_type": latent_type,
                                    "clean_up_videos": True, 
                                    "selected_loras": [],
                                    "resolutionW": resolutionW,
                                    "resolutionH": resolutionH,
                                    "lora_loaded_names": lora_names,
                                    "lora_values": []
                                }

                                def xy_plot_convert_values(type, value_textbox, value_dropdown):
                                    retVal = []
                                    if type[0] == "dropdown":
                                        retVal = value_dropdown
                                    elif type[0] == "textbox":
                                        retVal = xy_plot_parse_input(value_textbox)
                                    elif type[0] == "number":
                                        if type[1] == "int":
                                            retVal = [int(float(x)) for x in xy_plot_parse_input(value_textbox)]
                                        else:
                                            retVal = [float(x) for x in xy_plot_parse_input(value_textbox)]
                                    return retVal
                                prompt_replace_initial_values = {}
                                all_axis_values = {
                                    axis_x_switch+" -> X": xy_plot_convert_values(xy_plot_axis_options[axis_x_switch], axis_x_value_text, axis_x_value_dropdown)
                                }
                                if axis_x_switch == "Prompt replace":
                                    prompt_replace_initial_values["X"] = all_axis_values[axis_x_switch+" -> X"][0]
                                    if prompt_replace_initial_values["X"] not in base_generator_vars["prompt"]:
                                        return "Prompt for replacing in X axis not present in generation prompt", gr.update()
                                if axis_y_switch != "Nothing":
                                    all_axis_values[axis_y_switch+" -> Y"] = xy_plot_convert_values(xy_plot_axis_options[axis_y_switch], axis_y_value_text, axis_y_value_dropdown)
                                    if axis_y_switch == "Prompt replace":
                                        prompt_replace_initial_values["Y"] = all_axis_values[axis_y_switch+" -> Y"][0]
                                        if prompt_replace_initial_values["Y"] not in base_generator_vars["prompt"]:
                                            return "Prompt for replacing in Y axis not present in generation prompt", gr.update()
                                if axis_z_switch != "Nothing":
                                    all_axis_values[axis_z_switch+" -> Z"] = xy_plot_convert_values(xy_plot_axis_options[axis_z_switch], axis_z_value_text, axis_z_value_dropdown)
                                    if axis_z_switch == "Prompt replace":
                                        prompt_replace_initial_values["Z"] = all_axis_values[axis_z_switch+" -> Z"][0]
                                        if prompt_replace_initial_values["Z"] not in base_generator_vars["prompt"]:
                                            return "Prompt for replacing in Z axis not present in generation prompt", gr.update()

                                active_axes = list(all_axis_values.keys())
                                value_lists = [all_axis_values[axis] for axis in active_axes]
                                output_generator_vars = []

                                combintion_plot = itertools.product(*value_lists)
                                for combo in combintion_plot:
                                    vars_copy = base_generator_vars.copy()
                                    for axis, value in zip(active_axes, combo):
                                        splitted_axis_name = axis.split(" -> ")
                                        if splitted_axis_name[0] == "Prompt add":
                                            vars_copy[text_to_base_keys[splitted_axis_name[0]]] = vars_copy[text_to_base_keys[splitted_axis_name[0]]] + " " + str(value)
                                        elif splitted_axis_name[0] == "Prompt replace":
                                            orig_copy_prompt_text = vars_copy[text_to_base_keys[splitted_axis_name[0]]]
                                            vars_copy[text_to_base_keys[splitted_axis_name[0]]] = orig_copy_prompt_text.replace(prompt_replace_initial_values[splitted_axis_name[1]], str(value))
                                        else:
                                            vars_copy[text_to_base_keys[splitted_axis_name[0]]] = value
                                        vars_copy[splitted_axis_name[1]+"_axis_on_plot"] = str(value)
                                    output_generator_vars.append(xy_plot_convert_loras_text(vars_copy))
                                # print("----- BEFORE GENERATED VIDS VARS START -----")
                                # for v in output_generator_vars:
                                #     print(v)
                                # print("------ BEFORE GENERATED VIDS VARS END ------")

                                for i, v in enumerate(output_generator_vars):
                                    xy_plot_new_job = process_with_queue_update(
                                                v["model_type"], v["input_image"], v["input_video"], v["end_frame_image_original"], 
                                                v["end_frame_strength_original"], v["prompt"], v["n_prompt"],
                                                v["seed"], v["total_second_length"], v["latent_window_size"], v["steps"],
                                                v["cfg"], v["gs"], v["rs"], v["gpu_memory_preservation"],
                                                v["use_teacache"], v["teacache_num_steps"], v["teacache_rel_l1_thresh"], v["mp4_crf"], v["randomize_seed_checked"], False,
                                                v["blend_sections"], v["latent_type"], v["clean_up_videos"], v["selected_loras"],
                                                v["resolutionW"], v["resolutionH"], v["lora_loaded_names"], v["lora_values"]
                                            )
                                    output_generator_vars[i]["job_id"] = xy_plot_new_job[1]
                                # blah...
                                while True:
                                    xy_plot_ended_jobs = 0
                                    # outVrS = []
                                    for i, gen in enumerate(output_generator_vars):
                                        job = job_queue.get_job(gen["job_id"])
                                        if job.result != None and job.status == JobStatus.COMPLETED:
                                            output_generator_vars[i]["result"] = job.result
                                            xy_plot_ended_jobs += 1
                                        # outVrS.append([gen["job_id"], job.result, job.status])
                                    # print("Waiting")
                                    # for ktv in outVrS:
                                        # print(ktv)
                                    if xy_plot_ended_jobs == len(output_generator_vars):
                                        print("All jobs for XY plot done")
                                        break
                                    time.sleep(5)
                                # print("----- GENERATED VIDS VARS START -----")
                                # for v in output_generator_vars:
                                #     print(v)
                                # print("------ GENERATED VIDS VARS END ------")

                                # -------------------------- connect with settings --------------------------
                                output_dir = 'outputs'
                                # -------------------------- connect with settings --------------------------

                                # ---------------------------- what about linux? ----------------------------
                                fontfile = "C\\\\:/Windows/Fonts/arial.ttf"
                                # ---------------------------- what about linux? ----------------------------

                                timestamp_generation = generate_timestamp()
                                output_path = output_dir + "//" + timestamp_generation + "_grid_XY.mp4"
                                has_y = any('Y_axis_on_plot' in v for v in output_generator_vars)
                                # has_z = any('Z_axis_on_plot' in v for v in output_generator_vars) # not for now, please....

                                # maybe needed bucket size
                                # out_bucket_resH, out_bucket_resW = [128, 128]
                                # if input_image is not None:
                                #     H, W, _ = input_image.shape
                                #     out_bucket_resH, out_bucket_resW = find_nearest_bucket(H, W, resolution=resolutionW)
                                # else:
                                #     out_bucket_resH, out_bucket_resW = find_nearest_bucket(resolutionH, resolutionW, (resolutionW+resolutionH)/2)

                                x_labels = []
                                y_labels = []
                                grid = {}
                                for item in output_generator_vars:
                                    x = item['X_axis_on_plot']
                                    y = item.get('Y_axis_on_plot', 'single_row')  # fallback for single-row
                                    if x not in x_labels:
                                        x_labels.append(x)
                                    if y not in y_labels:
                                        y_labels.append(y)
                                    grid[(y, x)] = item['result']
                                inputs = []
                                filter_complex = ''
                                input_id = 0
                                row_filters = []
                                for y_index, y in enumerate(y_labels):
                                    col_filters = []
                                    for x_index, x in enumerate(x_labels):
                                        path = grid.get((y, x))
                                        if not path or not os.path.exists(path):
                                            raise ValueError(f"Missing or invalid video for X={x}, Y={y}")
                                        inputs.append(f"-i \"{path}\"")
                                        label_filters = []
                                        if y_index == 0:
                                            label_filters.append(
                                                f"drawtext=fontfile={fontfile}:text='{x}':x=(w-text_w)/2:y=2:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.5"
                                            )
                                        if x_index == 0 and has_y:
                                            label_filters.append(
                                                f"drawtext=fontfile={fontfile}:text='{y}':x=2:y=(h-text_h)/2:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.5"
                                            )
                                        if label_filters:
                                            filter_complex += f"[{input_id}:v]{','.join(label_filters)}[v{input_id}];"
                                            col_filters.append(f"[v{input_id}]")
                                        else:
                                            col_filters.append(f"[{input_id}:v]")
                                        input_id += 1
                                    if len(col_filters) == 1:
                                        row_filters.append(col_filters[0])
                                    else:
                                        row_label = f"row_{y_index}"
                                        filter_complex += f"{''.join(col_filters)}hstack=inputs={len(col_filters)}[{row_label}];"
                                        row_filters.append(f"[{row_label}]")
                                if len(row_filters) == 1:
                                    final_map = row_filters[0]
                                else:
                                    filter_complex += f"{''.join(row_filters)}vstack=inputs={len(row_filters)}[out]"
                                    final_map = "[out]"
                                ffmpeg_command = f"ffmpeg {' '.join(inputs)} -filter_complex \"{filter_complex}\" -map {final_map} -y \"{output_path}\""
                                # print("Running FFmpeg with labels:")
                                # print(ffmpeg_command)
                                subprocess.run(ffmpeg_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                return "", gr.update(value=output_path, visible=True)

                            with gr.Row():
                                xy_plot_model_type = gr.Radio(
                                    ["Original", "F1"], 
                                    label="Model Type", 
                                    value="F1",
                                    info="Select which model to use for generation"
                                )
                            with gr.Group():
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        xy_plot_input_image = gr.Image(
                                            sources='upload',
                                            type="numpy",
                                            label="Image (optional)",
                                            height=420,
                                            image_mode="RGB",
                                            elem_classes="contain-image"
                                        )
                                    with gr.Column(scale=1):
                                        xy_plot_end_frame_image_original = gr.Image(
                                            sources='upload',
                                            type="numpy",
                                            label="End Frame (Optional)", 
                                            height=420, 
                                            elem_classes="contain-image",
                                            image_mode="RGB",
                                            show_download_button=False,
                                            show_label=True,
                                            container=True
                                        )
                                with gr.Group():
                                    xy_plot_end_frame_strength_original = gr.Slider(
                                        label="End Frame Influence",
                                        minimum=0.05,
                                        maximum=1.0,
                                        value=1.0,
                                        step=0.05,
                                        info="Controls how strongly the end frame guides the generation. 1.0 is full influence."
                                    )
                            with gr.Accordion("Latent Image Options", open=False):
                                xy_plot_latent_type = gr.Dropdown(
                                    ["Black", "White", "Noise", "Green Screen"], 
                                    label="Latent Image", 
                                    value="Black", 
                                    info="Used as a starting point if no image is provided"
                                )
                            xy_plot_prompt = gr.Textbox(label="Prompt", value=default_prompt)
                            with gr.Accordion("Prompt Parameters", open=False):
                                xy_plot_blend_sections = gr.Slider(
                                    minimum=0, maximum=10, value=4, step=1,
                                    label="Number of sections to blend between prompts"
                                )
                            with gr.Accordion("Generation Parameters", open=True):
                                with gr.Row():
                                    xy_plot_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=5, step=1)
                                    xy_plot_total_second_length = gr.Slider(label="Video Length (Seconds)", minimum=0.1, maximum=120, value=1, step=0.1)
                                with gr.Group():
                                    with gr.Row("Resolution"):
                                        xy_plot_resolutionW = gr.Slider(
                                            label="Width", minimum=128, maximum=768, value=128, step=32, 
                                            info="Nearest valid width will be used."
                                        )
                                        xy_plot_resolutionH = gr.Slider(
                                            label="Height", minimum=128, maximum=768, value=128, step=32, 
                                            info="Nearest valid height will be used."
                                        )
                                    xy_plot_resolution_text = gr.Markdown(value="<div style='text-align:right; padding:5px 15px 5px 5px;'>Selected bucket for resolution: 128 x 128</div>", label="", show_label=False)
                                def xy_plot_on_input_image_change(img):
                                    if img is not None:
                                        return gr.update(info="Nearest valid bucket size will be used. Height will be adjusted automatically."), gr.update(visible=False)
                                    else:
                                        return gr.update(info="Nearest valid width will be used."), gr.update(visible=True)
                                xy_plot_input_image.change(fn=xy_plot_on_input_image_change, inputs=[xy_plot_input_image], outputs=[xy_plot_resolutionW, xy_plot_resolutionH])
                                def xy_plot_on_resolution_change(img, resolutionW, resolutionH):
                                    out_bucket_resH, out_bucket_resW = [128, 128]
                                    if img is not None:
                                        H, W, _ = img.shape
                                        out_bucket_resH, out_bucket_resW = find_nearest_bucket(H, W, resolution=resolutionW)
                                    else:
                                        out_bucket_resH, out_bucket_resW = find_nearest_bucket(resolutionH, resolutionW, (resolutionW+resolutionH)/2) # if resolutionW > resolutionH else resolutionH
                                    return gr.update(value=f"<div style='text-align:right; padding:5px 15px 5px 5px;'>Selected bucket for resolution: {out_bucket_resW} x {out_bucket_resH}</div>")
                                xy_plot_resolutionW.change(fn=xy_plot_on_resolution_change, inputs=[xy_plot_input_image, xy_plot_resolutionW, xy_plot_resolutionH], outputs=[xy_plot_resolution_text], show_progress="hidden")
                                xy_plot_resolutionH.change(fn=xy_plot_on_resolution_change, inputs=[xy_plot_input_image, xy_plot_resolutionW, xy_plot_resolutionH], outputs=[xy_plot_resolution_text], show_progress="hidden")
                                with gr.Row():
                                    xy_plot_seed = gr.Number(label="Seed", value=31337, precision=0)
                                    xy_plot_randomize_seed = gr.Checkbox(label="Randomize", value=False, info="Generate a new random seed for each job")
                            with gr.Accordion("Advanced Parameters", open=False):
                                with gr.Row("TeaCache"):
                                    xy_plot_use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                                    xy_plot_teacache_num_steps = gr.Slider(label="TeaCache steps", minimum=1, maximum=50, step=1, value=25, visible=True, info='How many intermediate sections to keep in the cache')
                                    xy_plot_teacache_rel_l1_thresh = gr.Slider(label="TeaCache rel_l1_thresh", minimum=0.01, maximum=1.0, step=0.01, value=0.15, visible=True, info='Relative L1 Threshold')
                                    xy_plot_use_teacache.change(lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled)), inputs=xy_plot_use_teacache, outputs=[xy_plot_teacache_num_steps, xy_plot_teacache_rel_l1_thresh])

                                xy_plot_latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True, info='Change at your own risk, very experimental')  # Should not change
                                xy_plot_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                                xy_plot_gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                                xy_plot_rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change
                                xy_plot_gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=1, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                            with gr.Accordion("Output Parameters", open=False):
                                xy_plot_mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                            with gr.Accordion("Plot Parameters", open=True):
                                def xy_plot_axis_change(updated_value_type):
                                    if xy_plot_axis_options[updated_value_type][0] == "textbox" or xy_plot_axis_options[updated_value_type][0] == "number":
                                        return gr.update(visible=True, value=xy_plot_axis_options[updated_value_type][2]), gr.update(visible=False, value=[], choices=[])
                                    elif xy_plot_axis_options[updated_value_type][0] == "dropdown":
                                        return gr.update(visible=False), gr.update(visible=True, value=xy_plot_axis_options[updated_value_type][2], choices=xy_plot_axis_options[updated_value_type][1])
                                    else:
                                        return gr.update(visible=False), gr.update(visible=False, value=[], choices=[])
                                with gr.Row():
                                    xy_plot_axis_x_switch = gr.Dropdown(label="X axis type for plotting", choices=list(xy_plot_axis_options.keys()))
                                    xy_plot_axis_x_value_text = gr.Textbox(label="X axis comma separated text", visible=False)
                                    xy_plot_axis_x_value_dropdown = gr.CheckboxGroup(label="X axis values", visible=False) #, multiselect=True)
                                    xy_plot_axis_x_switch.change(fn=xy_plot_axis_change, inputs=[xy_plot_axis_x_switch], outputs=[xy_plot_axis_x_value_text, xy_plot_axis_x_value_dropdown])
                                with gr.Row():
                                    xy_plot_axis_y_switch = gr.Dropdown(label="Y axis type for plotting", choices=list(xy_plot_axis_options.keys()))
                                    xy_plot_axis_y_value_text = gr.Textbox(label="Y axis comma separated text", visible=False)
                                    xy_plot_axis_y_value_dropdown = gr.CheckboxGroup(label="Y axis values", visible=False) #, multiselect=True)
                                    xy_plot_axis_y_switch.change(fn=xy_plot_axis_change, inputs=[xy_plot_axis_y_switch], outputs=[xy_plot_axis_y_value_text, xy_plot_axis_y_value_dropdown])
                                with gr.Row(visible=False): # not implemented Z axis
                                    xy_plot_axis_z_switch = gr.Dropdown(label="Z axis type for plotting", choices=list(xy_plot_axis_options.keys()))
                                    xy_plot_axis_z_value_text = gr.Textbox(label="Z axis comma separated text", visible=False)
                                    xy_plot_axis_z_value_dropdown = gr.CheckboxGroup(label="Z axis values", visible=False) #, multiselect=True)
                                    xy_plot_axis_z_switch.change(fn=xy_plot_axis_change, inputs=[xy_plot_axis_z_switch], outputs=[xy_plot_axis_z_value_text, xy_plot_axis_z_value_dropdown])

                            xy_plot_process_btn = gr.Button("Submit")
                            # xy_plot_result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=256, loop=True)
                            xy_plot_status = gr.HTML("")
                            xy_plot_output = gr.Video(autoplay=True, loop=True, sources=[], height=256, visible=False) # or Gallery, but return need value=[paths] instead of value=video
                            xy_plot_process_btn.click(fn=xy_plot_process, inputs=[xy_plot_model_type, xy_plot_input_image, xy_plot_end_frame_image_original,
                                                                            xy_plot_end_frame_strength_original, xy_plot_latent_type, 
                                                                            xy_plot_prompt, xy_plot_blend_sections, xy_plot_steps, xy_plot_total_second_length, 
                                                                            xy_plot_resolutionW, xy_plot_resolutionH, xy_plot_seed, xy_plot_randomize_seed, 
                                                                            xy_plot_use_teacache, xy_plot_teacache_num_steps, xy_plot_teacache_rel_l1_thresh, 
                                                                            xy_plot_latent_window_size, xy_plot_cfg, xy_plot_gs, xy_plot_rs, 
                                                                            xy_plot_gpu_memory_preservation, xy_plot_mp4_crf, 
                                                                            xy_plot_axis_x_switch, xy_plot_axis_x_value_text, xy_plot_axis_x_value_dropdown, 
                                                                            xy_plot_axis_y_switch, xy_plot_axis_y_value_text, xy_plot_axis_y_value_dropdown, 
                                                                            xy_plot_axis_z_switch, xy_plot_axis_z_value_text, xy_plot_axis_z_value_dropdown
                                                                            ], outputs=[xy_plot_status, xy_plot_output])
                        with gr.Group(visible=True) as standard_generation_group:    # Default visibility: True because "Original" model is not "Video"
                            with gr.Group(visible=True) as image_input_group: # This group now only contains the start frame image
                                with gr.Row():
                                    with gr.Column(scale=1): # Start Frame Image Column
                                        input_image = gr.Image(
                                            sources='upload',
                                            type="numpy",
                                            label="Start Frame (optional)",
                                            elem_classes="contain-image",
                                            image_mode="RGB",
                                            show_download_button=False,
                                            show_label=True, # Keep label for clarity
                                            container=True
                                        )
                            
                            with gr.Group(visible=False) as video_input_group:
                                input_video = gr.Video(
                                    sources='upload',
                                    label="Video Input",
                                    height=420,
                                    show_label=True
                                )
                                combine_with_source = gr.Checkbox(
                                    label="Combine with source video",
                                    value=True,
                                    info="If checked, the source video will be combined with the generated video",
                                    interactive=True
                                )
                                num_cleaned_frames = gr.Slider(label="Number of Context Frames (Adherence to Video)", minimum=2, maximum=10, value=5, step=1, interactive=True, info="Expensive. Retain more video details. Reduce if memory issues or motion too restricted (jumpcut, ignoring prompt, still).")

                            
                            # End Frame Image Input
                            # Initial visibility is False, controlled by update_input_visibility
                            with gr.Column(scale=1, visible=False) as end_frame_group_original:
                                end_frame_image_original = gr.Image(
                                    sources='upload',
                                    type="numpy",
                                    label="End Frame (Optional)", 
                                    elem_classes="contain-image",
                                    image_mode="RGB",
                                    show_download_button=False,
                                    show_label=True,
                                    container=True
                                )
                            
                            # End Frame Influence slider
                            # Initial visibility is False, controlled by update_input_visibility
                            with gr.Group(visible=False) as end_frame_slider_group:
                                end_frame_strength_original = gr.Slider(
                                    label="End Frame Influence",
                                    minimum=0.05,
                                    maximum=1.0,
                                    value=1.0,
                                    step=0.05,
                                    info="Controls how strongly the end frame guides the generation. 1.0 is full influence."
                                )

                            with gr.Accordion("Latent Image Options", open=False):
                                latent_type = gr.Dropdown(
                                    ["Black", "White", "Noise", "Green Screen"], label="Latent Image", value="Black", info="Used as a starting point if no image is provided"
                                )

                            prompt = gr.Textbox(label="Prompt", value=default_prompt)

                            with gr.Accordion("Prompt Parameters", open=False):
                                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=True)  # Make visible for both models

                                blend_sections = gr.Slider(
                                    minimum=0, maximum=10, value=4, step=1,
                                    label="Number of sections to blend between prompts"
                                )
                            with gr.Accordion("Generation Parameters", open=True):
                                with gr.Row():
                                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                                    total_second_length = gr.Slider(label="Video Length (Seconds)", minimum=1, maximum=120, value=6, step=0.1)
                                with gr.Group():
                                    with gr.Row("Resolution"):
                                        resolutionW = gr.Slider(
                                            label="Width", minimum=128, maximum=768, value=640, step=32, 
                                            info="Nearest valid width will be used."
                                        )
                                        resolutionH = gr.Slider(
                                            label="Height", minimum=128, maximum=768, value=640, step=32, 
                                            info="Nearest valid height will be used."
                                        )
                                    resolution_text = gr.Markdown(value="<div style='text-align:right; padding:5px 15px 5px 5px;'>Selected bucket for resolution: 640 x 640</div>", label="", show_label=False)
                                def on_input_image_change(img):
                                    if img is not None:
                                        return gr.update(info="Nearest valid bucket size will be used. Height will be adjusted automatically."), gr.update(visible=False)
                                    else:
                                        return gr.update(info="Nearest valid width will be used."), gr.update(visible=True)
                                input_image.change(fn=on_input_image_change, inputs=[input_image], outputs=[resolutionW, resolutionH])
                                def on_resolution_change(img, resolutionW, resolutionH):
                                    out_bucket_resH, out_bucket_resW = [640, 640]
                                    if img is not None:
                                        H, W, _ = img.shape
                                        out_bucket_resH, out_bucket_resW = find_nearest_bucket(H, W, resolution=resolutionW)
                                    else:
                                        out_bucket_resH, out_bucket_resW = find_nearest_bucket(resolutionH, resolutionW, (resolutionW+resolutionH)/2) # if resolutionW > resolutionH else resolutionH
                                    return gr.update(value=f"<div style='text-align:right; padding:5px 15px 5px 5px;'>Selected bucket for resolution: {out_bucket_resW} x {out_bucket_resH}</div>")
                                resolutionW.change(fn=on_resolution_change, inputs=[input_image, resolutionW, resolutionH], outputs=[resolution_text], show_progress="hidden")
                                resolutionH.change(fn=on_resolution_change, inputs=[input_image, resolutionW, resolutionH], outputs=[resolution_text], show_progress="hidden")
                                with gr.Row("LoRAs"):
                                    lora_selector = gr.Dropdown(
                                        choices=lora_names,
                                        label="Select LoRAs to Load",
                                        multiselect=True,
                                        value=[],
                                        info="Select one or more LoRAs to use for this job"
                                    )
                                    lora_names_states = gr.State(lora_names)
                                    lora_sliders = {}
                                    for lora in lora_names:
                                        lora_sliders[lora] = gr.Slider(
                                            minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                                            label=f"{lora} Weight", visible=False, interactive=True
                                        )

                                with gr.Row("Metadata"):
                                    json_upload = gr.File(
                                        label="Upload Metadata JSON (optional)",
                                        file_types=[".json"],
                                        type="filepath",
                                        height=100,
                                    )
                                with gr.Row("TeaCache"):
                                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                                    teacache_num_steps = gr.Slider(label="TeaCache steps", minimum=1, maximum=50, step=1, value=25, visible=True, info='How many intermediate sections to keep in the cache')
                                    teacache_rel_l1_thresh = gr.Slider(label="TeaCache rel_l1_thresh", minimum=0.01, maximum=1.0, step=0.01, value=0.15, visible=True, info='Relative L1 Threshold')
                                    use_teacache.change(lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled)), inputs=use_teacache, outputs=[teacache_num_steps, teacache_rel_l1_thresh])

                                with gr.Row():
                                    seed = gr.Number(label="Seed", value=2500, precision=0)
                                    randomize_seed = gr.Checkbox(label="Randomize", value=True, info="Generate a new random seed for each job")

                            with gr.Accordion("Advanced Parameters", open=False):
                                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True, info='Change at your own risk, very experimental')  # Should not change
                                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                    with gr.Column():
                        preview_image = gr.Image(
                            label="Next Latents", 
                            height=150, 
                            visible=True, 
                            type="numpy", 
                            interactive=False,
                            elem_classes="contain-image",
                            image_mode="RGB"
                        )
                        result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=256, loop=True)
                        progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                        progress_bar = gr.HTML('', elem_classes='no-generating-animation')

                        with gr.Row():
                            current_job_id = gr.Textbox(label="Current Job ID", visible=True, interactive=True)
                            start_button = gr.Button(value="Add to Queue", variant="primary", elem_id="toolbar-add-to-queue-btn")
                            end_button = gr.Button(value="Cancel Current Job", interactive=True, visible=False)

           

            with gr.Tab("Queue"):
                with gr.Row():
                    with gr.Column():
                        # Create a container for the queue status
                        with gr.Row():
                            queue_status = gr.DataFrame(
                                headers=["Job ID", "Type", "Status", "Created", "Started", "Completed", "Elapsed", "Preview"], 
                                datatype=["str", "str", "str", "str", "str", "str", "str", "html"], 
                                label="Job Queue"
                            )
                        with gr.Row():
                            refresh_button = gr.Button("Refresh Queue")
                            clear_queue_button = gr.Button("Cancel Queue", variant="stop")
                            clear_complete_button = gr.Button("Clear Complete", variant="secondary")
                            load_queue_button = gr.Button("Load Queue")
                            queue_export_button = gr.Button("Queue Export")
                            import_queue_file = gr.File(
                                label="Import Queue",
                                file_types=[".json", ".zip"],
                                type="filepath",
                                visible=True
                            )
                            # Connect the refresh button (Moved inside 'with block')
                            refresh_button.click(
                                fn=update_queue_status_fn, # Use the function passed in
                                inputs=[],
                                outputs=[queue_status]
                            )
                            
                            # Function to clear all jobs in the queue
                            def clear_all_jobs():
                                try:
                                    # Use the clear_queue method from VideoJobQueue
                                    cancelled_count = job_queue.clear_queue()
                                    print(f"Cleared {cancelled_count} jobs from the queue")
                                    
                                    # Refresh the queue display with the current state
                                    return update_queue_status_fn()
                                except Exception as e:
                                    import traceback
                                    print(f"Error in clear_all_jobs: {e}")
                                    traceback.print_exc()
                                    # Return empty list on error to avoid UI hanging
                                    return []
                            
                            # Connect the clear queue button
                            clear_queue_button.click(
                                fn=clear_all_jobs,
                                inputs=[],
                                outputs=[queue_status]
                            )
                            
                            # Function to clear completed and cancelled jobs
                            def clear_completed_jobs():
                                try:
                                    # Use the clear_completed_jobs method from VideoJobQueue
                                    removed_count = job_queue.clear_completed_jobs()
                                    print(f"Removed {removed_count} completed/cancelled jobs from the queue")
                                    
                                    # Refresh the queue display with the current state
                                    return update_queue_status_fn()
                                except Exception as e:
                                    import traceback
                                    print(f"Error in clear_completed_jobs: {e}")
                                    traceback.print_exc()
                                    # Return empty list on error to avoid UI hanging
                                    return []
                            
                            # Connect the clear complete button
                            clear_complete_button.click(
                                fn=clear_completed_jobs,
                                inputs=[],
                                outputs=[queue_status]
                            )
                            
                            # Function to load queue from queue.json
                            def load_queue_from_json():
                                try:
                                    # Use the load_queue_from_json method from VideoJobQueue
                                    loaded_count = job_queue.load_queue_from_json()
                                    print(f"Loaded {loaded_count} jobs from queue.json")
                                    
                                    # Refresh the queue display with the current state
                                    return update_queue_status_fn()
                                except Exception as e:
                                    import traceback
                                    print(f"Error loading queue from JSON: {e}")
                                    traceback.print_exc()
                                    # Return empty list on error to avoid UI hanging
                                    return []
                            
                            # Function to import queue from a custom JSON file
                            def import_queue_from_file(file_path):
                                if not file_path:
                                    return update_queue_status_fn()
                                    
                                try:
                                    # Use the load_queue_from_json method from VideoJobQueue with the provided file path
                                    loaded_count = job_queue.load_queue_from_json(file_path)
                                    print(f"Loaded {loaded_count} jobs from {file_path}")
                                    
                                    # Refresh the queue display with the current state
                                    return update_queue_status_fn()
                                except Exception as e:
                                    import traceback
                                    print(f"Error importing queue from file: {e}")
                                    traceback.print_exc()
                                    # Return empty list on error to avoid UI hanging
                                    return []
                            
                            # Connect the load queue button
                            load_queue_button.click(
                                fn=load_queue_from_json,
                                inputs=[],
                                outputs=[queue_status]
                            )
                            
                            # Function to export queue to a zip file
                            def export_queue_to_zip():
                                try:
                                    # Use the export_queue_to_zip method from VideoJobQueue
                                    zip_path = job_queue.export_queue_to_zip()
                                    if zip_path and os.path.exists(zip_path):
                                        print(f"Queue exported to {zip_path}")
                                        return update_queue_status_fn()
                                    else:
                                        print("Failed to export queue to zip")
                                        return update_queue_status_fn()
                                except Exception as e:
                                    import traceback
                                    print(f"Error exporting queue to zip: {e}")
                                    traceback.print_exc()
                                    # Return empty list on error to avoid UI hanging
                                    return []
                            
                            # Connect the queue export button
                            queue_export_button.click(
                                fn=export_queue_to_zip,
                                inputs=[],
                                outputs=[queue_status]
                            )
                            
                            # Connect the import queue file selector
                            import_queue_file.change(
                                fn=import_queue_from_file,
                                inputs=[import_queue_file],
                                outputs=[queue_status]
                            )
                        # Add documentation accordion
                        with gr.Accordion("Queue Documentation", open=True):
                            gr.Markdown("""
                            ## Queue Tab Documentation
                            
                            The Queue tab allows you to manage your generation jobs. Here's what each button does:
                            
                            ### Queue Controls
                            
                            - **Refresh Queue**: Updates the queue status display with the current state of all jobs.
                            
                            - **Cancel Queue**: Cancels all pending jobs in the queue. This will not affect jobs that are currently running.
                            
                            - **Clear Complete**: Removes all completed and cancelled jobs from the queue, keeping only pending and running jobs.
                            
                            - **Load Queue**: Loads the queue from the default `queue.json` file in the application directory. This is useful if you've restarted the application and want to restore your previous queue state.
                            
                            - **Queue Export**: Packages the current queue configuration (`queue.json`) and all associated images (from the `queue_images` directory) into a single zip file named `queue_export.zip`. This makes it easy to back up your queue or transfer it to another installation.
                            
                            - **Import Queue**: Allows you to import a queue from either:
                              - A JSON file containing queue configuration
                              - A ZIP file containing both queue configuration and associated images (created using Queue Export)
                            
                            ### Queue Table
                            
                            The table displays all jobs in the queue with the following information:
                            
                            - **Job ID**: Unique identifier for each job
                            - **Type**: The generation model type used for the job
                            - **Status**: Current status of the job (pending, running, completed, failed, cancelled)
                            - **Created**: When the job was added to the queue
                            - **Started**: When the job began processing
                            - **Completed**: When the job finished processing
                            - **Elapsed**: How long the job took to process
                            - **Preview**: Thumbnail preview of the job's input image or latent type
                            """)
                            
                        # Create a container for thumbnails (kept for potential future use, though not displayed in DataFrame)
                        with gr.Row():
                            thumbnail_container = gr.Column()
                            thumbnail_container.elem_classes = ["thumbnail-container"]

                        # Add CSS for thumbnails
            with gr.Tab("Outputs", id="outputs_tab"): # Ensure 'id' is present for tab switching
                outputDirectory_video = settings.get("output_dir", settings.default_settings['output_dir'])
                outputDirectory_metadata = settings.get("metadata_dir", settings.default_settings['metadata_dir'])
                def get_gallery_items():
                    items = []
                    for f in os.listdir(outputDirectory_metadata):
                        if f.endswith(".png"):
                            prefix = os.path.splitext(f)[0]
                            latest_video = get_latest_video_version(prefix)
                            if latest_video:
                                video_path = os.path.join(outputDirectory_video, latest_video)
                                mtime = os.path.getmtime(video_path)
                                preview_path = os.path.join(outputDirectory_metadata, f)
                                items.append((preview_path, prefix, mtime))
                    items.sort(key=lambda x: x[2], reverse=True)
                    return [(i[0], i[1]) for i in items]
                def get_latest_video_version(prefix):
                    max_number = -1
                    selected_file = None
                    for f in os.listdir(outputDirectory_video):
                        if f.startswith(prefix + "_") and f.endswith(".mp4"):
                            num = int(f.replace(prefix + "_", '').replace(".mp4", ''))
                            if num > max_number:
                                max_number = num
                                selected_file = f
                    return selected_file
                # load_video_and_info_from_prefix now also returns button visibility
                def load_video_and_info_from_prefix(prefix):
                    video_file = get_latest_video_version(prefix)
                    json_path = os.path.join(outputDirectory_metadata, prefix) + ".json"
                    
                    if not video_file or not os.path.exists(os.path.join(outputDirectory_video, video_file)) or not os.path.exists(json_path):
                        # If video or info not found, button should be hidden
                        return None, "Video or JSON not found.", gr.update(visible=False) 

                    video_path = os.path.join(outputDirectory_video, video_file)
                    info_content = {"description": "no info"}
                    if os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as f:
                            info_content = json.load(f)
                    # If video and info found, button should be visible
                    return video_path, json.dumps(info_content, indent=2, ensure_ascii=False), gr.update(visible=True)

                gallery_items_state = gr.State(get_gallery_items())
                selected_original_video_path_state = gr.State(None) # Holds the ORIGINAL, UNPROCESSED path
                with gr.Row():
                    with gr.Column(scale=2):
                        thumbs = gr.Gallery(
                            # value=[i[0] for i in get_gallery_items()],
                            columns=[4],
                            allow_preview=False,
                            object_fit="cover",
                            height="auto"
                        )
                        refresh_button = gr.Button("Update")
                    with gr.Column(scale=5):
                        video_out = gr.Video(sources=[], autoplay=True, loop=True, visible=False)
                    with gr.Column(scale=1):
                        info_out = gr.Textbox(label="Generation info", visible=False)
                        send_to_toolbox_btn = gr.Button("➡️ Send to Post-processing", visible=False)  # Added new send_to_toolbox_btn
                    def refresh_gallery():
                        new_items = get_gallery_items()
                        return gr.update(value=[i[0] for i in new_items]), new_items
                    refresh_button.click(fn=refresh_gallery, outputs=[thumbs, gallery_items_state])
                    
                    # MODIFIED: on_select now handles visibility of the new button
                    def on_select(evt: gr.SelectData, gallery_items):
                        if evt.index is None or not gallery_items or evt.index >= len(gallery_items):
                            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None

                        prefix = gallery_items[evt.index][1]
                        # original_video_path is e.g., "outputs/my_actual_video.mp4"
                        original_video_path, info_string, button_visibility_update = load_video_and_info_from_prefix(prefix)

                        # Determine visibility for video and info based on whether video_path was found
                        video_out_update = gr.update(value=original_video_path, visible=bool(original_video_path))
                        info_out_update = gr.update(value=info_string, visible=bool(original_video_path))

                        # IMPORTANT: Store the ORIGINAL, UNPROCESSED path in the gr.State
                        return video_out_update, info_out_update, button_visibility_update, original_video_path

                    thumbs.select(
                        fn=on_select,
                        inputs=[gallery_items_state],
                        outputs=[video_out, info_out, send_to_toolbox_btn, selected_original_video_path_state] # Output original path to State
                    )
            with gr.Tab("Post-processing", id="toolbox_tab"):          
                # Call the function from toolbox_app.py to build the Toolbox UI
                # The toolbox_ui_layout (e.g., a gr.Column) is automatically placed here.                
                toolbox_ui_layout, tb_target_video_input = tb_create_video_toolbox_ui()
                
            with gr.Tab("Settings"):
                with gr.Row():
                    with gr.Column():
                        save_metadata = gr.Checkbox(
                            label="Save Metadata", 
                            info="Save to JSON file", 
                            value=settings.get("save_metadata", 6),
                        )
                        gpu_memory_preservation = gr.Slider(
                            label="GPU Inference Preserved Memory (GB) (larger means slower)",
                            minimum=1,
                            maximum=128,
                            step=0.1,
                            value=settings.get("gpu_memory_preservation", 6),
                            info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed."
                        )
                        mp4_crf = gr.Slider(
                            label="MP4 Compression",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=settings.get("mp4_crf", 16),
                            info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs."
                        )
                        clean_up_videos = gr.Checkbox(
                            label="Clean up video files",
                            value=settings.get("clean_up_videos", True),
                            info="If checked, only the final video will be kept after generation."
                        )
                        cleanup_temp_folder = gr.Checkbox(
                            label="Clean up temp folder after generation",
                            visible=False,
                            value=settings.get("cleanup_temp_folder", True),
                            info="If checked, temporary files will be cleaned up after each generation."
                        )
                        
                        with gr.Accordion("System Prompt", open=False):
                            override_system_prompt = gr.Checkbox(
                                label="Override System Prompt",
                                value=settings.get("override_system_prompt", False),
                                info="If checked, the system prompt template below will be used instead of the default one."
                            )
                            system_prompt_template = gr.Textbox(
                                label="System Prompt Template",
                                value=settings.get("system_prompt_template", "{\"template\": \"<|start_header_id|>system<|end_header_id|>\\n\\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{}<|eot_id|>\", \"crop_start\": 95}"),
                                lines=10,
                                info="System prompt template used for video generation. Must be a valid JSON or Python dictionary string with 'template' and 'crop_start' keys. Example: {\"template\": \"your template here\", \"crop_start\": 95}"
                            )
                        output_dir = gr.Textbox(
                            label="Output Directory",
                            value=settings.get("output_dir"),
                            placeholder="Path to save generated videos"
                        )
                        metadata_dir = gr.Textbox(
                            label="Metadata Directory",
                            value=settings.get("metadata_dir"),
                            placeholder="Path to save metadata files"
                        )
                        lora_dir = gr.Textbox(
                            label="LoRA Directory",
                            value=settings.get("lora_dir"),
                            placeholder="Path to LoRA models"
                        )
                        gradio_temp_dir = gr.Textbox(label="Gradio Temporary Directory", value=settings.get("gradio_temp_dir"))
                        auto_save = gr.Checkbox(
                            label="Auto-save settings",
                            value=settings.get("auto_save_settings", True)
                        )
                        # Add Gradio Theme Dropdown
                        gradio_themes = ["default", "base", "soft", "glass", "mono", "huggingface"]
                        theme_dropdown = gr.Dropdown(
                            label="Theme",
                            choices=gradio_themes,
                            value=settings.get("gradio_theme", "soft"),
                            info="Select the Gradio UI theme. Requires restart."
                        )
                        save_btn = gr.Button("Save Settings")
                        cleanup_btn = gr.Button("Clean Up Temporary Files")
                        status = gr.HTML("")
                        cleanup_output = gr.Textbox(label="Cleanup Status", interactive=False)

                        def save_settings(save_metadata, gpu_memory_preservation, mp4_crf, clean_up_videos, cleanup_temp_folder, override_system_prompt_value, system_prompt_template_value, output_dir, metadata_dir, lora_dir, gradio_temp_dir, auto_save, selected_theme):
                            try:
                                # Save the system prompt template as is, without trying to parse it
                                # The hunyuan.py file will handle parsing it when needed
                                processed_template = system_prompt_template_value
                                
                                settings.save_settings(
                                    save_metadata=save_metadata,
                                    gpu_memory_preservation=gpu_memory_preservation,
                                    mp4_crf=mp4_crf,
                                    clean_up_videos=clean_up_videos,
                                    cleanup_temp_folder=cleanup_temp_folder,
                                    override_system_prompt=override_system_prompt_value,
                                    system_prompt_template=processed_template,
                                    output_dir=output_dir,
                                    metadata_dir=metadata_dir,
                                    lora_dir=lora_dir,
                                    gradio_temp_dir=gradio_temp_dir,
                                    auto_save_settings=auto_save,
                                    gradio_theme=selected_theme
                                )
                                return "<p style='color:green;'>Settings saved successfully! Restart required for theme change.</p>"
                            except Exception as e:
                                return f"<p style='color:red;'>Error saving settings: {str(e)}</p>"

                        save_btn.click(
                            fn=save_settings,
                            inputs=[save_metadata, gpu_memory_preservation, mp4_crf, clean_up_videos, cleanup_temp_folder, override_system_prompt, system_prompt_template, output_dir, metadata_dir, lora_dir, gradio_temp_dir, auto_save, theme_dropdown],
                            outputs=[status]
                        )

                        def cleanup_temp_files():
                            """Clean up temporary files and folders in the Gradio temp directory"""
                            temp_dir = settings.get("gradio_temp_dir")
                            if not temp_dir or not os.path.exists(temp_dir):
                                return "No temporary directory found or directory does not exist."
                            
                            try:
                                # Get all items in the temp directory
                                items = os.listdir(temp_dir)
                                removed_count = 0
                                print(f"Finding items in {temp_dir}")
                                for item in items:
                                    item_path = os.path.join(temp_dir, item)
                                    try:
                                        if os.path.isfile(item_path) or os.path.islink(item_path):
                                            print(f"Removing {item_path}")
                                            os.remove(item_path)
                                            removed_count += 1
                                        elif os.path.isdir(item_path):
                                            print(f"Removing directory {item_path}")
                                            shutil.rmtree(item_path)
                                            removed_count += 1
                                    except Exception as e:
                                        print(f"Error removing {item_path}: {e}")
                                
                                return f"Cleaned up {removed_count} temporary files/folders."
                            except Exception as e:
                                return f"Error cleaning up temporary files: {str(e)}"

        # --- Event Handlers and Connections (Now correctly indented) ---

        # Connect the main process function (wrapper for adding to queue)
        def process_with_queue_update(model_type_arg, *args):
            # Extract all arguments (ensure order matches inputs lists)
            # The order here MUST match the order in the `ips` list.
            # RT_BORG: Global settings gpu_memory_preservation, mp4_crf, save_metadata removed from direct args.
            (input_image_arg,
             input_video_arg,
             end_frame_image_original_arg,
             end_frame_strength_original_arg,
             prompt_text_arg,
             n_prompt_arg,
             seed_arg,
             total_second_length_arg,
             latent_window_size_arg,
             steps_arg,
             cfg_arg, 
             gs_arg,
             rs_arg,
             use_teacache_arg,
             teacache_num_steps_arg,
             teacache_rel_l1_thresh_arg,
             randomize_seed_arg,
             save_metadata_checked_arg, # NEW: Extract save_metadata_checked parameter
             blend_sections_arg,
             latent_type_arg,
             clean_up_videos_arg, # UI checkbox from Generate tab
             selected_loras_arg,
             resolutionW_arg, resolutionH_arg,
             combine_with_source_arg, # <<<< Correctly unpacked
             num_cleaned_frames_arg,  # <<<< Correctly unpacked
             lora_names_states_arg,   # This is from lora_names_states (gr.State)
             *lora_slider_values_tuple # Remaining args are LoRA slider values
            ) = args
            # DO NOT parse the prompt here. Parsing happens once in the worker.

            # Determine the model type to send to the backend
            backend_model_type = model_type_arg # model_type_arg is the UI selection
            if model_type_arg == "Video with Endframe":
                backend_model_type = "Video" # The backend "Video" model_type handles with and without endframe

            # Use the appropriate input based on model type
            is_ui_video_model = (model_type_arg == "Video" or model_type_arg == "Video with Endframe" or model_type_arg == "Video F1")
            input_data = input_video_arg if is_ui_video_model else input_image_arg

            # Define actual end_frame params to pass to backend
            actual_end_frame_image_for_backend = None
            actual_end_frame_strength_for_backend = 1.0  # Default strength

            if model_type_arg == "Original with Endframe" or model_type_arg == "F1 with Endframe" or model_type_arg == "Video with Endframe":
                actual_end_frame_image_for_backend = end_frame_image_original_arg
                actual_end_frame_strength_for_backend = end_frame_strength_original_arg

            # Get the input video path for Video model
            input_image_path = None
            if is_ui_video_model and input_video_arg is not None:
                # For Video models, input_video contains the path to the video file
                input_image_path = input_video_arg

            # Use the current seed value as is for this job
            # Call the process function with all arguments
            # Pass the backend_model_type and the ORIGINAL prompt_text string to the backend process function
            result = process_fn(backend_model_type, input_data, actual_end_frame_image_for_backend, actual_end_frame_strength_for_backend,
                                prompt_text_arg, n_prompt_arg, seed_arg, total_second_length_arg,
                                latent_window_size_arg, steps_arg, cfg_arg, gs_arg, rs_arg,
                                use_teacache_arg, teacache_num_steps_arg, teacache_rel_l1_thresh_arg,
                                blend_sections_arg, latent_type_arg, clean_up_videos_arg, # clean_up_videos_arg is from UI
                                selected_loras_arg, resolutionW_arg, resolutionH_arg, 
                                input_image_path, 
                                combine_with_source_arg,
                                num_cleaned_frames_arg,
                                lora_names_states_arg,
                                *lora_slider_values_tuple,
                                save_metadata_checked=save_metadata_checked_arg # NEW: Pass save_metadata_checked parameter
                               )
            # If randomize_seed is checked, generate a new random seed for the next job
            new_seed_value = None
            if randomize_seed_arg:
                new_seed_value = random.randint(0, 21474)
                print(f"Generated new seed for next job: {new_seed_value}")

            # Create a button update that will be applied after the job is added to the queue
            # This ensures the button text is reset after the queue's JSON has been saved
            button_update = gr.update(value="Add to Queue", interactive=True)
            
            # If a job ID was created, automatically start monitoring it and update queue
            if result and result[1]:  # Check if job_id exists in results
                job_id = result[1]
                queue_status_data = update_queue_status_fn()

                # Add the new seed value to the results if randomize is checked
                if new_seed_value is not None:
                    # Use result[6] directly for end_button to preserve its value
                    return [result[0], job_id, result[2], result[3], result[4], result[5], result[6], queue_status_data, new_seed_value]
                else:
                    # Use result[6] directly for end_button to preserve its value
                    return [result[0], job_id, result[2], result[3], result[4], result[5], result[6], queue_status_data, gr.update()]

            # If no job ID was created, still return the new seed if randomize is checked
            if new_seed_value is not None:
                # Make sure to preserve the end_button update from result[6]
                return [result[0], result[1], result[2], result[3], result[4], result[5], result[6], update_queue_status_fn(), new_seed_value]
            else:
                # Make sure to preserve the end_button update from result[6]
                return [result[0], result[1], result[2], result[3], result[4], result[5], result[6], update_queue_status_fn(), gr.update()]

        # Custom end process function that ensures the queue is updated and changes button text
        def end_process_with_update():
            queue_status_data = end_process_fn()
            
            # Don't try to get the new job ID immediately after cancellation
            # The monitor_job function will handle the transition to the next job
            
            # Change the cancel button text to "Cancelling..." and make it non-interactive
            # This ensures the button stays in this state until the job is fully cancelled
            return queue_status_data, gr.update(value="Cancelling...", interactive=False), gr.update()

        # MODIFIED handle_send_video_to_toolbox:
        def handle_send_video_to_toolbox(original_path_from_state): # Input is now the original path from gr.State
            print(f"Button clicked. Sending ORIGINAL video path (from State) to Post-processing: {original_path_from_state}")

            if original_path_from_state and isinstance(original_path_from_state, str) and os.path.exists(original_path_from_state):
                # tb_target_video_input will now process the ORIGINAL path (e.g., "outputs/my_actual_video.mp4").
                return gr.update(value=original_path_from_state), gr.update(selected="toolbox_tab")
            else:
                print(f"No valid original video path from State to send. Path: {original_path_from_state}")
                return gr.update(), gr.update()

        send_to_toolbox_btn.click(
            fn=handle_send_video_to_toolbox,
            inputs=[selected_original_video_path_state], # INPUT IS NOW THE gr.State holding the ORIGINAL path
            outputs=[
                tb_target_video_input, # This is tb_input_video_component from toolbox_app.py
                main_tabs_component
            ]
        )
        
        # --- Inputs Lists ---
        # --- Inputs for all models ---
        ips = [
            input_image,                # Corresponds to input_image_arg
            input_video,                # Corresponds to input_video_arg
            end_frame_image_original,   # Corresponds to end_frame_image_original_arg
            end_frame_strength_original,# Corresponds to end_frame_strength_original_arg
            prompt,                     # Corresponds to prompt_text_arg
            n_prompt,                   # Corresponds to n_prompt_arg
            seed,                       # Corresponds to seed_arg
            total_second_length,        # Corresponds to total_second_length_arg
            latent_window_size,         # Corresponds to latent_window_size_arg
            steps,                      # Corresponds to steps_arg
            cfg,                        # Corresponds to cfg_arg
            gs,                         # Corresponds to gs_arg
            rs,                         # Corresponds to rs_arg
            use_teacache,               # Corresponds to use_teacache_arg
            teacache_num_steps,         # Corresponds to teacache_num_steps_arg
            teacache_rel_l1_thresh,     # Corresponds to teacache_rel_l1_thresh_arg
            randomize_seed,             # Corresponds to randomize_seed_arg
            blend_sections,             # Corresponds to blend_sections_arg
            latent_type,                # Corresponds to latent_type_arg
            clean_up_videos,            # Corresponds to clean_up_videos_arg (UI checkbox)
            lora_selector,              # Corresponds to selected_loras_arg
            resolutionW,                # Corresponds to resolutionW_arg
            resolutionH,                # Corresponds to resolutionH_arg
            combine_with_source,        # Corresponds to combine_with_source_arg
            num_cleaned_frames,         # Corresponds to num_cleaned_frames_arg
            lora_names_states           # Corresponds to lora_names_states_arg
        ]
        # Add LoRA sliders to the input list
        ips.extend([lora_sliders[lora] for lora in lora_names])


        # --- Connect Buttons ---
        def handle_start_button(selected_model, *args):
            if selected_model == "XY Plot":
                # For XY Plot, call the xy_plot_process function
                status, video = generate_tab_xy_plot_process()
                if status:
                    # If there was an error, display it
                    return gr.update(value=None), gr.update(), gr.update(), gr.update(value=status), gr.update(), gr.update(value="Add to Queue", interactive=True), gr.update(), gr.update(), gr.update()
                else:
                    # If successful, display the result video
                    return gr.update(value=video), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value="Add to Queue", interactive=True), gr.update(), gr.update(), gr.update()
            else:
                # For other model types, use the regular process function
                # First update the button to show "Adding..." and disable it
                return process_with_queue_update(selected_model, *args)
                
        # Function to update button state before processing
        def update_button_before_processing(selected_model, *args):
            # First update the button to show "Adding..." and disable it
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value="Adding...", interactive=False), gr.update(), gr.update(), gr.update()
        
        # Connect the start button to first update its state
        start_button.click(
            fn=update_button_before_processing,
            inputs=[model_type] + ips,
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button, queue_status, seed]
        ).then(
            # Then process the job
            fn=handle_start_button,
            inputs=[model_type] + ips,
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button, queue_status, seed]
        )

        # Connect the end button to cancel the current job and update the queue
        end_button.click(
            fn=end_process_with_update,
            outputs=[queue_status, end_button, current_job_id]
        )


        #putting this here for now because this file is way too big
        def on_model_type_change(selected_model):
            if selected_model == "XY Plot":
                # If 'XY Plot' is selected, make standard_generation_group invisible
                return (
                    gr.update(visible=False),  # standard_generation_group
                    gr.update(visible=True)   # xy_group
                )
            else:
                # For any other model type, make standard_generation_group visible
                 return (
                    gr.update(visible=True),  # standard_generation_group
                    gr.update(visible=False)   # xy_group
                )

        # Model change listener
        model_type.change(
            fn=on_model_type_change,
            inputs=model_type,
            outputs=[standard_generation_group, xy_group]
        )

        

        # --- Connect Monitoring ---
        # Auto-monitor the current job when job_id changes
        # Monitor original tab
        current_job_id.change(
            fn=monitor_fn,
            inputs=[current_job_id],
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button]
        )
        
        # Auto-check for current job on page load
        def check_for_current_job():
            # This function will be called when the interface loads
            # It will check if there's a current job in the queue and update the UI
            with job_queue.lock:
                current_job = job_queue.current_job
                if current_job:
                    # Return all the necessary information to update the preview windows
                    job_id = current_job.id
                    result = current_job.result
                    preview = current_job.progress_data.get('preview') if current_job.progress_data else None
                    desc = current_job.progress_data.get('desc', '') if current_job.progress_data else ''
                    html = current_job.progress_data.get('html', '') if current_job.progress_data else ''
                    
                    # Also trigger the monitor_job function to start monitoring this job
                    print(f"Auto-check found current job {job_id}, triggering monitor_job")
                    return job_id, result, preview, desc, html
                return None, None, None, '', ''
                
        # Connect the auto-check function to the interface load event
        block.load(
            fn=check_for_current_job,
            inputs=[],
            outputs=[current_job_id, result_video, preview_image, progress_desc, progress_bar]
        )

        cleanup_btn.click(
            fn=cleanup_temp_files,
            outputs=[cleanup_output]
        )


        # --- Connect Queue Refresh ---
        # Function to update queue stats
        def update_stats():
            # Get queue status data
            queue_status_data = update_queue_status_fn()
            
            # Get queue statistics for the toolbar display
            jobs = job_queue.get_all_jobs()
            
            # Count jobs by status
            pending_count = 0
            running_count = 0
            completed_count = 0
            
            for job in jobs:
                if hasattr(job, 'status'):
                    status = str(job.status)
                    if status == "JobStatus.PENDING":
                        pending_count += 1
                    elif status == "JobStatus.RUNNING":
                        running_count += 1
                    elif status == "JobStatus.COMPLETED":
                        completed_count += 1
            
            # Format the queue stats display text
            queue_stats_text = f"<p style='margin:0;color:white;' class='toolbar-text'>Queue: {pending_count} | Running: {running_count} | Completed: {completed_count}</p>"
            
            return queue_status_data, queue_stats_text
        
        refresh_stats_btn.click(
            fn=update_stats,
            inputs=None,
            outputs=[queue_status, queue_stats_display]
        )

        # Set up auto-refresh for queue status
        # Instead of using a timer with 'every' parameter, we'll use the queue refresh button
        # and rely on manual refreshes. The user can click the refresh button in the toolbar
        # to update the stats.

        # --- Connect LoRA UI ---
        # Function to update slider visibility based on selection
        def update_lora_sliders(selected_loras):
            updates = []
            # Need to handle potential missing keys if lora_names changes dynamically
            # For now, assume lora_names passed to create_interface is static
            for lora in lora_names:
                 updates.append(gr.update(visible=(lora in selected_loras)))
            # Ensure the output list matches the number of sliders defined
            num_sliders = len(lora_sliders)
            return updates[:num_sliders] # Return only updates for existing sliders

        # Connect the dropdown to the sliders
        lora_selector.change(
            fn=update_lora_sliders,
            inputs=[lora_selector],
            outputs=[lora_sliders[lora] for lora in lora_names] # Assumes lora_sliders keys match lora_names
        )

        # --- Auto-refresh for Toolbar System Stats Monitor (Timer) ---
        main_toolbar_system_stats_timer = gr.Timer(2, active=True) 
        
        main_toolbar_system_stats_timer.tick(
            fn=tb_get_formatted_toolbar_stats, # Function imported from toolbox_app.py
            inputs=None, 
            outputs=[ # Target the Textbox components
                toolbar_ram_display_component,
                toolbar_vram_display_component,
                toolbar_gpu_display_component 
            ]
        )
        
        # --- Connect Metadata Loading ---
        # Function to load metadata from JSON file
        def load_metadata_from_json(json_path):
            if not json_path:
                # Return updates for all potentially affected components
                num_orig_sliders = len(lora_sliders)
                return [gr.update()] * (4 + num_orig_sliders)  # Updated to include total_second_length and end_frame_strength

            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                prompt_val = metadata.get('prompt')
                seed_val = metadata.get('seed')
                total_second_length_val = metadata.get('total_second_length')  # Get total_second_length
                end_frame_strength_val = metadata.get('end_frame_strength') # Load end_frame_strength if present in metadata
                
                # Check for LoRA values in metadata
                lora_weights = metadata.get('loras', {}) # Changed key to 'loras' based on studio.py worker

                print(f"Loaded metadata from JSON: {json_path}")
                print(f"Prompt: {prompt_val}, Seed: {seed_val}, Total Second Length: {total_second_length_val}")

                # Update the UI components
                updates = [
                    gr.update(value=prompt_val) if prompt_val else gr.update(),
                    gr.update(value=seed_val) if seed_val is not None else gr.update(),
                    gr.update(value=total_second_length_val) if total_second_length_val is not None else gr.update(),  # Add total_second_length update
                    gr.update(value=end_frame_strength_val) if end_frame_strength_val is not None else gr.update(),  # Add update for end_frame_strength_original slider
                ]

                # Update LoRA sliders if they exist in metadata
                for lora in lora_names:
                    if lora in lora_weights:
                        updates.append(gr.update(value=lora_weights[lora]))
                    else:
                        updates.append(gr.update()) # No change if LoRA not in metadata

                # Ensure the number of updates matches the number of outputs
                num_orig_sliders = len(lora_sliders)
                return updates[:4 + num_orig_sliders] # Return updates for prompt, seed, total_second_length, end_frame_strength, and sliders

            except Exception as e:
                print(f"Error loading metadata: {e}")
                num_orig_sliders = len(lora_sliders)
                return [gr.update()] * (4 + num_orig_sliders)  # Updated to include total_second_length and end_frame_strength


        # Connect JSON metadata loader for Original tab
        json_upload.change(
            fn=load_metadata_from_json,
            inputs=[json_upload],
            outputs=[prompt, seed, total_second_length, end_frame_strength_original] + [lora_sliders[lora] for lora in lora_names]
        )


        # --- Helper Functions (defined within create_interface scope if needed by handlers) ---
        # Function to get queue statistics
        def get_queue_stats():
            try:
                # Get all jobs from the queue
                jobs = job_queue.get_all_jobs()

                # Count jobs by status
                status_counts = {
                    "QUEUED": 0,
                    "RUNNING": 0,
                    "COMPLETED": 0,
                    "FAILED": 0,
                    "CANCELLED": 0
                }

                for job in jobs:
                    if hasattr(job, 'status'):
                        status = str(job.status) # Use str() for safety
                        if status in status_counts:
                            status_counts[status] += 1

                # Format the display text
                stats_text = f"Queue: {status_counts['QUEUED']} | Running: {status_counts['RUNNING']} | Completed: {status_counts['COMPLETED']} | Failed: {status_counts['FAILED']} | Cancelled: {status_counts['CANCELLED']}"

                return f"<p style='margin:0;color:white;'>{stats_text}</p>"

            except Exception as e:
                print(f"Error getting queue stats: {e}")
                return "<p style='margin:0;color:white;'>Error loading queue stats</p>"

        # Add footer with social links
        with gr.Row(elem_id="footer"):
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="text-align: center; padding: 20px; color: #666;">
                    <div style="margin-top: 10px;">
                        <a href="https://patreon.com/Colinu" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;" class="footer-patreon">
                            <i class="fab fa-patreon"></i>Support on Patreon
                        </a>
                        <a href="https://discord.gg/MtuM7gFJ3V" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;">
                            <i class="fab fa-discord"></i> Discord
                        </a>
                        <a href="https://github.com/colinurbs/FramePack-Studio" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;">
                            <i class="fab fa-github"></i> GitHub
                        </a>
                    </div>
                </div>
                """)

        # Add CSS for footer

        return block

# --- Top-level Helper Functions (Used by Gradio callbacks, must be defined outside create_interface) ---

def format_queue_status(jobs):
    """Format job data for display in the queue status table"""
    rows = []
    for job in jobs:
        created = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.created_at)) if job.created_at else ""
        started = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.started_at)) if job.started_at else ""
        completed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.completed_at)) if job.completed_at else ""

        # Calculate elapsed time
        elapsed_time = ""
        if job.started_at:
            if job.completed_at:
                start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                complete_datetime = datetime.datetime.fromtimestamp(job.completed_at)
                elapsed_seconds = (complete_datetime - start_datetime).total_seconds()
                elapsed_time = f"{elapsed_seconds:.2f}s"
            else:
                # For running jobs, calculate elapsed time from now
                start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                current_datetime = datetime.datetime.now()
                elapsed_seconds = (current_datetime - start_datetime).total_seconds()
                elapsed_time = f"{elapsed_seconds:.2f}s (running)"

        # Get generation type from job data
        generation_type = getattr(job, 'generation_type', 'Original')

        # Get thumbnail from job data and format it as HTML for display
        thumbnail = getattr(job, 'thumbnail', None)
        thumbnail_html = f'<img src="{thumbnail}" width="64" height="64" style="object-fit: contain;">' if thumbnail else ""

        rows.append([
            job.id[:6] + '...',
            generation_type,
            job.status.value,
            created,
            started,
            completed,
            elapsed_time,
            thumbnail_html  # Add formatted thumbnail HTML to row data
        ])
    return rows

# Create the queue status update function (wrapper around format_queue_status)
def update_queue_status_with_thumbnails(): # Function name is now slightly misleading, but keep for now to avoid breaking clicks
    # This function is likely called by the refresh button and potentially the timer
    # It needs access to the job_queue object
    # Assuming job_queue is accessible globally or passed appropriately
    # For now, let's assume it's globally accessible as defined in studio.py
    # If not, this needs adjustment based on how job_queue is managed.
    try:
        # Need access to the global job_queue instance from studio.py
        # This might require restructuring or passing job_queue differently.
        # For now, assuming it's accessible (this might fail if run standalone)
        from __main__ import job_queue # Attempt to import from main script scope

        jobs = job_queue.get_all_jobs()
        for job in jobs:
            if job.status == JobStatus.PENDING:
                job.queue_position = job_queue.get_queue_position(job.id)

        if job_queue.current_job:
            job_queue.current_job.status = JobStatus.RUNNING

        return format_queue_status(jobs)
    except ImportError:
        print("Error: Could not import job_queue. Queue status update might fail.")
        return [] # Return empty list on error
    except Exception as e:
        print(f"Error updating queue status: {e}")
        return []
