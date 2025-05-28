"""
Metadata utilities for FramePack Studio.
This module provides functions for generating and saving metadata.
"""

import os
import json
import time
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

def get_placeholder_color(model_type):
    """
    Get the placeholder image color for a specific model type.
    
    Args:
        model_type: The model type string
        
    Returns:
        RGB tuple for the placeholder image color
    """
    # Define color mapping for different model types
    color_map = {
        "Original": (0, 0, 0),           # Black
        "F1": (0, 0, 128),               # Blue
        "Video": (0, 128, 0),            # Green
        "XY Plot": (128, 128, 0),        # Yellow
        "F1 with Endframe": (0, 128, 128),  # Teal
        "Original with Endframe": (128, 0, 128),  # Purple
    }
    
    # Return the color for the model type, or black as default
    return color_map.get(model_type, (0, 0, 0))

def create_metadata(job_params, job_id, settings):
    """
    Create metadata for the job.
    
    Args:
        job_params: Dictionary of job parameters
        job_id: The job ID
        settings: Dictionary of settings
        
    Returns:
        Metadata dictionary
    """
    if not settings.get("save_metadata"):
        return None
    
    metadata_dir = settings.get("metadata_dir")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Get model type and determine placeholder image color
    model_type = job_params.get('model_type', "Original")
    placeholder_color = get_placeholder_color(model_type)
    
    # Create a placeholder image
    height = job_params.get('height', 640)
    width = job_params.get('width', 640)
    
    # Use resolutionH and resolutionW if height and width are not available
    if not height:
        height = job_params.get('resolutionH', 640)
    if not width:
        width = job_params.get('resolutionW', 640)
        
    placeholder_img = Image.new('RGB', (width, height), placeholder_color)
    
    # Add XY plot parameters to the image if applicable
    if model_type == "XY Plot":
        x_param = job_params.get('x_param', '')
        y_param = job_params.get('y_param', '')
        x_values = job_params.get('x_values', [])
        y_values = job_params.get('y_values', [])
        
        draw = ImageDraw.Draw(placeholder_img)
        try:
            # Try to use a system font
            font = ImageFont.truetype("Arial", 20)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
        
        text = f"X: {x_param} - {x_values}\nY: {y_param} - {y_values}"
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    
    # Create PNG metadata
    metadata = PngInfo()
    metadata.add_text("prompt", job_params.get('prompt_text', ''))
    metadata.add_text("seed", str(job_params.get('seed', 0)))
    
    # Add model-specific metadata to PNG
    if model_type == "XY Plot":
        metadata.add_text("x_param", job_params.get('x_param', ''))
        metadata.add_text("y_param", job_params.get('y_param', ''))
    
    # Only save placeholder image with metadata if called from worker.py
    # This is determined by checking if the call stack includes 'worker.py'
    import traceback
    call_stack = traceback.extract_stack()
    is_called_from_worker = any('worker.py' in frame.filename for frame in call_stack)
    
    if is_called_from_worker:
        # Save placeholder image with metadata
        placeholder_img.save(os.path.join(metadata_dir, f'{job_id}.png'), pnginfo=metadata)
    
    # Create comprehensive JSON metadata with all possible parameters
    metadata_dict = {
        # Common parameters
        "prompt": job_params.get('prompt_text', ''),
        "negative_prompt": job_params.get('n_prompt', ''),
        "seed": job_params.get('seed', 0),
        "steps": job_params.get('steps', 25),
        "cfg": job_params.get('cfg', 1.0),
        "gs": job_params.get('gs', 10.0),
        "rs": job_params.get('rs', 0.0),
        "latent_type": job_params.get('latent_type', 'Black'),
        "timestamp": time.time(),
        "resolutionW": job_params.get('resolutionW', 640),
        "resolutionH": job_params.get('resolutionH', 640),
        "model_type": model_type,
        "has_input_image": job_params.get('has_input_image', False),
        "input_image_path": job_params.get('input_image_path', None),
        
        # Video-related parameters
        "total_second_length": job_params.get('total_second_length', 6),
        "blend_sections": job_params.get('blend_sections', 4),
        "latent_window_size": job_params.get('latent_window_size', 9),
        
        # Endframe-related parameters
        "end_frame_strength": job_params.get('end_frame_strength', None),
        "end_frame_image_path": job_params.get('end_frame_image_path', None),
        "end_frame_used": True if job_params.get('end_frame_image') is not None else False,
        
        # Video input-related parameters
        "input_video": os.path.basename(job_params.get('input_video_path', '')) if job_params.get('input_video_path') else None,
        "video_path": job_params.get('input_image') if model_type == "Video" else None,
        
        # XY Plot-related parameters
        "x_param": job_params.get('x_param', None),
        "y_param": job_params.get('y_param', None),
        "x_values": job_params.get('x_values', None),
        "y_values": job_params.get('y_values', None),
        
        # Tea cache parameters
        "use_teacache": job_params.get('use_teacache', False),
        "teacache_num_steps": job_params.get('teacache_num_steps', 0),
        "teacache_rel_l1_thresh": job_params.get('teacache_rel_l1_thresh', 0.0)
    }
    
    # Add LoRA information if present
    selected_loras = job_params.get('selected_loras', [])
    lora_values = job_params.get('lora_values', [])
    lora_loaded_names = job_params.get('lora_loaded_names', [])
    
    if selected_loras and len(selected_loras) > 0:
        lora_data = {}
        for lora_name in selected_loras:
            try:
                idx = lora_loaded_names.index(lora_name)
                weight = lora_values[idx] if lora_values and idx < len(lora_values) else 1.0
                if isinstance(weight, list):
                    weight_value = weight[0] if weight and len(weight) > 0 else 1.0
                else:
                    weight_value = weight
                lora_data[lora_name] = float(weight_value)
            except ValueError:
                lora_data[lora_name] = 1.0
        metadata_dict["loras"] = lora_data
    else:
        metadata_dict["loras"] = {}
    
    # Only save JSON metadata if called from worker.py
    if is_called_from_worker:
        # Save JSON metadata
        with open(os.path.join(metadata_dir, f'{job_id}.json'), 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    return metadata_dict
