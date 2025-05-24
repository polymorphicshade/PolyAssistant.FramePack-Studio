"""
Base pipeline class for FramePack Studio.
All pipeline implementations should inherit from this class.
"""

import os
import json
import time
from PIL import Image
from PIL.PngImagePlugin import PngInfo

class BasePipeline:
    """Base class for all pipeline implementations."""
    
    def __init__(self, settings):
        """
        Initialize the pipeline with settings.
        
        Args:
            settings: Dictionary of settings for the pipeline
        """
        self.settings = settings
    
    def prepare_parameters(self, job_params):
        """
        Prepare parameters for the job.
        
        Args:
            job_params: Dictionary of job parameters
            
        Returns:
            Processed parameters dictionary
        """
        # Default implementation just returns the parameters as-is
        return job_params
    
    def validate_parameters(self, job_params):
        """
        Validate parameters for the job.
        
        Args:
            job_params: Dictionary of job parameters
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Default implementation assumes all parameters are valid
        return True, None
    
    def preprocess_inputs(self, job_params):
        """
        Preprocess input images/videos for the job.
        
        Args:
            job_params: Dictionary of job parameters
            
        Returns:
            Processed inputs dictionary
        """
        # Default implementation returns an empty dictionary
        return {}
    
    def handle_results(self, job_params, result):
        """
        Handle the results of the job.
        
        Args:
            job_params: Dictionary of job parameters
            result: The result of the job
            
        Returns:
            Processed result
        """
        # Default implementation just returns the result as-is
        return result
    
    def create_metadata(self, job_params, job_id):
        """
        Create metadata for the job.
        
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
        
        # Create a placeholder image
        height = job_params.get('height', 640)
        width = job_params.get('width', 640)
        placeholder_img = Image.new('RGB', (width, height), (0, 0, 0))
        
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
            "model_type": job_params.get('model_type', "Original")
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
