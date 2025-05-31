"""
Metadata utilities for FramePack Studio.
This module provides functions for generating and saving metadata.
"""

import os
import json
import time
import traceback # Moved to top
import numpy as np # Added
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

# Function to save the starting image with comprehensive metadata
def save_job_start_image(job_params, job_id, settings):
    """
    Saves the job's starting input image to the output directory with comprehensive metadata.
    This is intended to be called early in the job processing and is the ONLY place metadata should be saved.
    """
    # Get output directory from settings or job_params
    output_dir_path = job_params.get("output_dir") or settings.get("output_dir")
    metadata_dir_path = job_params.get("metadata_dir") or settings.get("metadata_dir")
    
    if not output_dir_path:
        print(f"[JOB_START_IMG_ERROR] No output directory found in job_params or settings")
        return False
        
    # Ensure directories exist
    os.makedirs(output_dir_path, exist_ok=True)
    os.makedirs(metadata_dir_path, exist_ok=True)

    actual_start_image_target_path = os.path.join(output_dir_path, f'{job_id}.png')
    actual_input_image_np = job_params.get('input_image')

    print(f"[JOB_START_IMG_DEBUG] Job ID: {job_id} - Attempting to save starting image.")
    print(f"[JOB_START_IMG_DEBUG] Target path: {actual_start_image_target_path}")
    print(f"[JOB_START_IMG_DEBUG] Type of job_params['input_image']: {type(actual_input_image_np)}")

    # Create comprehensive metadata dictionary
    metadata_dict = create_metadata(job_params, job_id, settings)
    
    # Save JSON metadata with the same job_id
    json_metadata_path = os.path.join(metadata_dir_path, f'{job_id}.json')
    print(f"[METADATA_DEBUG] Saving JSON metadata to: {json_metadata_path}")
    try:
        with open(json_metadata_path, 'w') as f:
            import json
            json.dump(metadata_dict, f, indent=2)
        print(f"[METADATA_SUCCESS] Saved JSON metadata to {json_metadata_path}")
    except Exception as e:
        print(f"[METADATA_ERROR] Error saving JSON metadata to {json_metadata_path}: {e}")
        traceback.print_exc()

    # Save the input image if it's a numpy array
    if actual_input_image_np is not None and isinstance(actual_input_image_np, np.ndarray):
        print(f"[JOB_START_IMG_DEBUG] input_image is a NumPy array. Shape: {actual_input_image_np.shape}, Dtype: {actual_input_image_np.dtype}.")
        try:
            # Create PNG metadata
            png_metadata = PngInfo()
            png_metadata.add_text("prompt", job_params.get('prompt_text', ''))
            png_metadata.add_text("seed", str(job_params.get('seed', 0)))
            png_metadata.add_text("model_type", job_params.get('model_type', "Unknown"))
            
            # Add more metadata fields
            for key, value in metadata_dict.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    png_metadata.add_text(key, str(value))

            # Convert image if needed
            image_to_save_np = actual_input_image_np
            if actual_input_image_np.dtype != np.uint8:
                print(f"[JOB_START_IMG_DEBUG] Warning: input_image dtype is {actual_input_image_np.dtype}, attempting conversion to uint8.")
                if actual_input_image_np.max() <= 1.0 and actual_input_image_np.min() >= -1.0 and actual_input_image_np.dtype in [np.float32, np.float64]:
                     image_to_save_np = ((actual_input_image_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
                     print(f"[JOB_START_IMG_DEBUG] Converted float image from [-1,1] range to uint8.")
                elif actual_input_image_np.max() <= 1.0 and actual_input_image_np.min() >= 0.0 and actual_input_image_np.dtype in [np.float32, np.float64]:
                     image_to_save_np = (actual_input_image_np * 255.0).clip(0,255).astype(np.uint8)
                     print(f"[JOB_START_IMG_DEBUG] Converted float image from [0,1] range to uint8.")
                else:
                     image_to_save_np = actual_input_image_np.clip(0, 255).astype(np.uint8)
                     print(f"[JOB_START_IMG_DEBUG] Clipped/casted image to uint8 from dtype {actual_input_image_np.dtype}.")
            else:
                print(f"[JOB_START_IMG_DEBUG] input_image is already uint8. No conversion needed.")
            
            # Save the image with metadata
            start_image_pil = Image.fromarray(image_to_save_np)
            start_image_pil.save(actual_start_image_target_path, pnginfo=png_metadata)
            print(f"[JOB_START_IMG_SUCCESS] Saved actual starting image to {actual_start_image_target_path}")
            return True # Indicate success
        except Exception as e:
            print(f"[JOB_START_IMG_ERROR] Error saving actual starting image to {actual_start_image_target_path}: {e}")
            traceback.print_exc()
    elif actual_input_image_np is None:
        print(f"[JOB_START_IMG_DEBUG] job_params['input_image'] is None. Cannot save starting image.")
    else: # Not None, but not np.ndarray
        print(f"[JOB_START_IMG_DEBUG] job_params['input_image'] is not a NumPy array (type: {type(actual_input_image_np)}). Cannot save starting image.")
    
    return False # Indicate failure or inability to save

def create_metadata(job_params, job_id, settings, save_placeholder=False):
    """
    Create metadata for the job.
    
    Args:
        job_params: Dictionary of job parameters
        job_id: The job ID
        settings: Dictionary of settings
        save_placeholder: Whether to save the placeholder image (default: False)
        
    Returns:
        Metadata dictionary
    """
    if not settings.get("save_metadata"):
        return None
    
    metadata_dir_path = settings.get("metadata_dir")
    output_dir_path = settings.get("output_dir")
    os.makedirs(metadata_dir_path, exist_ok=True)
    os.makedirs(output_dir_path, exist_ok=True) # Ensure output_dir also exists
    
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
    
    # Determine end_frame_used value safely (avoiding NumPy array boolean ambiguity)
    end_frame_image = job_params.get('end_frame_image')
    end_frame_used = False
    if end_frame_image is not None:
        if isinstance(end_frame_image, np.ndarray):
            end_frame_used = end_frame_image.any()  # True if any element is non-zero
        else:
            end_frame_used = True
    
    # Create comprehensive JSON metadata with all possible parameters
    # This is created before file saving logic that might use it (e.g. JSON dump)
    # but PngInfo 'metadata' is used for images.
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
        "num_cleaned_frames": job_params.get('num_cleaned_frames', 5),
        
        # Endframe-related parameters
        "end_frame_strength": job_params.get('end_frame_strength', None),
        "end_frame_image_path": job_params.get('end_frame_image_path', None),
        "end_frame_used": str(end_frame_used),
        
        # Video input-related parameters
        "input_video": os.path.basename(job_params.get('input_image', '')) if job_params.get('input_image') is not None and model_type == "Video" else None,
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
                # Fix for NumPy array boolean ambiguity
                has_lora_values = lora_values is not None and len(lora_values) > 0
                weight = lora_values[idx] if has_lora_values and idx < len(lora_values) else 1.0
                
                # Handle different types of weight values
                if isinstance(weight, np.ndarray):
                    # Convert NumPy array to a scalar value
                    weight_value = float(weight.item()) if weight.size == 1 else float(weight.mean())
                elif isinstance(weight, list):
                    # Handle list type weights
                    has_items = weight is not None and len(weight) > 0
                    weight_value = float(weight[0]) if has_items else 1.0
                else:
                    # Handle scalar weights
                    weight_value = float(weight) if weight is not None else 1.0
                
                lora_data[lora_name] = weight_value
                print(f"[METADATA_DEBUG] Saved LoRA '{lora_name}' with weight {weight_value}")
            except ValueError:
                lora_data[lora_name] = 1.0
                print(f"[METADATA_DEBUG] LoRA '{lora_name}' not found in loaded names, using default weight 1.0")
            except Exception as e:
                lora_data[lora_name] = 1.0
                print(f"[METADATA_ERROR] Error processing LoRA '{lora_name}': {e}")
                traceback.print_exc()
        
        metadata_dict["loras"] = lora_data
        print(f"[METADATA_DEBUG] Saved LoRA data: {lora_data}")
    else:
        metadata_dict["loras"] = {}
        print("[METADATA_DEBUG] No LoRAs selected, saving empty loras dictionary")

    # This function now only creates the metadata dictionary without saving files
    # The actual saving is done by save_job_start_image() at the beginning of the generation process
    # This prevents duplicate metadata files from being created
    
    # For backward compatibility, we still create the placeholder image
    # and save it if explicitly requested
    placeholder_target_path = os.path.join(metadata_dir_path, f'{job_id}.png')
    print(f"[METADATA_DEBUG] Job ID: {job_id} - Created metadata dictionary")
    
    # Save the placeholder image if requested
    if save_placeholder:
        try:
            placeholder_img.save(placeholder_target_path, pnginfo=metadata)
            print(f"[METADATA_SUCCESS] Saved placeholder image to {placeholder_target_path}")
        except Exception as e:
            print(f"[METADATA_ERROR] Error saving placeholder image to {placeholder_target_path}: {e}")
            traceback.print_exc()
        
    return metadata_dict
