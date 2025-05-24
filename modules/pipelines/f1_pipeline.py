"""
F1 pipeline class for FramePack Studio.
This pipeline handles the "F1" model type.
"""

import os
import time
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers_helper.utils import resize_and_center_crop
from diffusers_helper.bucket_tools import find_nearest_bucket
from .base_pipeline import BasePipeline

class F1Pipeline(BasePipeline):
    """Pipeline for F1 generation type."""
    
    def prepare_parameters(self, job_params):
        """
        Prepare parameters for the F1 generation job.
        
        Args:
            job_params: Dictionary of job parameters
            
        Returns:
            Processed parameters dictionary
        """
        processed_params = job_params.copy()
        
        # Ensure we have the correct model type
        processed_params['model_type'] = "F1"
        
        return processed_params
    
    def validate_parameters(self, job_params):
        """
        Validate parameters for the F1 generation job.
        
        Args:
            job_params: Dictionary of job parameters
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for required parameters
        required_params = ['prompt_text', 'seed', 'total_second_length', 'steps']
        for param in required_params:
            if param not in job_params:
                return False, f"Missing required parameter: {param}"
        
        # Validate numeric parameters
        if job_params.get('total_second_length', 0) <= 0:
            return False, "Video length must be greater than 0"
        
        if job_params.get('steps', 0) <= 0:
            return False, "Steps must be greater than 0"
        
        return True, None
    
    def preprocess_inputs(self, job_params):
        """
        Preprocess input images for the F1 generation type.
        
        Args:
            job_params: Dictionary of job parameters
            
        Returns:
            Processed inputs dictionary
        """
        processed_inputs = {}
        
        # Process input image if provided
        input_image = job_params.get('input_image')
        if input_image is not None:
            # Get resolution parameters
            resolutionW = job_params.get('resolutionW', 640)
            resolutionH = job_params.get('resolutionH', 640)
            
            # Find nearest bucket size
            if job_params.get('has_input_image', True):
                # If we have an input image, use its dimensions to find the nearest bucket
                H, W, _ = input_image.shape
                height, width = find_nearest_bucket(H, W, resolution=resolutionW)
            else:
                # Otherwise, use the provided resolution parameters
                height, width = find_nearest_bucket(resolutionH, resolutionW, (resolutionW+resolutionH)/2)
            
            # Resize and center crop the input image
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            
            # Store the processed image and dimensions
            processed_inputs['input_image'] = input_image_np
            processed_inputs['height'] = height
            processed_inputs['width'] = width
        else:
            # If no input image, create a blank image based on latent_type
            resolutionW = job_params.get('resolutionW', 640)
            resolutionH = job_params.get('resolutionH', 640)
            height, width = find_nearest_bucket(resolutionH, resolutionW, (resolutionW+resolutionH)/2)
            
            latent_type = job_params.get('latent_type', 'Black')
            if latent_type == "White":
                # Create a white image
                input_image_np = np.ones((height, width, 3), dtype=np.uint8) * 255
            elif latent_type == "Noise":
                # Create a noise image
                input_image_np = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            elif latent_type == "Green Screen":
                # Create a green screen image with standard chroma key green (0, 177, 64)
                input_image_np = np.zeros((height, width, 3), dtype=np.uint8)
                input_image_np[:, :, 1] = 177  # Green channel
                input_image_np[:, :, 2] = 64   # Blue channel
                # Red channel remains 0
            else:  # Default to "Black" or any other value
                # Create a black image
                input_image_np = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Store the processed image and dimensions
            processed_inputs['input_image'] = input_image_np
            processed_inputs['height'] = height
            processed_inputs['width'] = width
        
        return processed_inputs
    
    def handle_results(self, job_params, result):
        """
        Handle the results of the F1 generation.
        
        Args:
            job_params: The job parameters
            result: The generation result
            
        Returns:
            Processed result
        """
        # For F1 generation, we just return the result as-is
        return result
    
    def create_metadata(self, job_params, job_id):
        """
        Create metadata for the F1 generation.
        
        Args:
            job_params: Dictionary of job parameters
            job_id: The job ID
            
        Returns:
            Metadata dictionary
        """
        if not self.settings.get("save_metadata"):
            return None
        
        metadata_dir = self.settings.get("metadata_dir")
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Create a placeholder image for the F1 model
        height = job_params.get('height', 640)
        width = job_params.get('width', 640)
        placeholder_img = Image.new('RGB', (width, height), (0, 0, 128))  # Blue for F1
        
        # Create PNG metadata
        metadata = PngInfo()
        metadata.add_text("prompt", job_params.get('prompt_text', ''))
        metadata.add_text("seed", str(job_params.get('seed', 0)))
        
        # Save placeholder image with metadata
        placeholder_img.save(os.path.join(metadata_dir, f'{job_id}.png'), pnginfo=metadata)
        
        # Create JSON metadata
        metadata_dict = {
            "prompt": job_params.get('prompt_text', ''),
            "seed": job_params.get('seed', 0),
            "total_second_length": job_params.get('total_second_length', 6),
            "steps": job_params.get('steps', 25),
            "cfg": job_params.get('cfg', 1.0),
            "gs": job_params.get('gs', 10.0),
            "rs": job_params.get('rs', 0.0),
            "latent_type": job_params.get('latent_type', 'Black'),
            "blend_sections": job_params.get('blend_sections', 4),
            "latent_window_size": job_params.get('latent_window_size', 9),
            "timestamp": time.time(),
            "resolutionW": job_params.get('resolutionW', 640),
            "resolutionH": job_params.get('resolutionH', 640),
            "model_type": "F1"
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
        
        # Save JSON metadata
        with open(os.path.join(metadata_dir, f'{job_id}.json'), 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        return metadata_dict
