"""
XY Plot pipeline class for FramePack Studio.
This pipeline handles the "XY Plot" model type.
"""

import os
import time
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
from diffusers_helper.utils import resize_and_center_crop
from diffusers_helper.bucket_tools import find_nearest_bucket
from .base_pipeline import BasePipeline

class XYPlotPipeline(BasePipeline):
    """Pipeline for XY Plot generation type."""
    
    def prepare_parameters(self, job_params):
        """
        Prepare parameters for the XY Plot generation job.
        
        Args:
            job_params: Dictionary of job parameters
            
        Returns:
            Processed parameters dictionary
        """
        processed_params = job_params.copy()
        
        # Ensure we have the correct model type
        processed_params['model_type'] = "XY Plot"
        
        return processed_params
    
    def validate_parameters(self, job_params):
        """
        Validate parameters for the XY Plot generation job.
        
        Args:
            job_params: Dictionary of job parameters
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for required parameters
        required_params = ['prompt_text', 'seed', 'steps']
        for param in required_params:
            if param not in job_params:
                return False, f"Missing required parameter: {param}"
        
        # Validate numeric parameters
        if job_params.get('steps', 0) <= 0:
            return False, "Steps must be greater than 0"
        
        # Check for XY plot parameters
        if not job_params.get('x_param') or not job_params.get('y_param'):
            return False, "X and Y parameters are required for XY Plot"
        
        if not job_params.get('x_values') or not job_params.get('y_values'):
            return False, "X and Y values are required for XY Plot"
        
        return True, None
    
    def preprocess_inputs(self, job_params):
        """
        Preprocess input images for the XY Plot generation type.
        
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
        
        # Process XY plot parameters
        x_param = job_params.get('x_param')
        y_param = job_params.get('y_param')
        x_values = job_params.get('x_values', [])
        y_values = job_params.get('y_values', [])
        
        # Store the XY plot parameters
        processed_inputs['x_param'] = x_param
        processed_inputs['y_param'] = y_param
        processed_inputs['x_values'] = x_values
        processed_inputs['y_values'] = y_values
        
        return processed_inputs
    
    def handle_results(self, job_params, result):
        """
        Handle the results of the XY Plot generation.
        
        Args:
            job_params: The job parameters
            result: The generation result
            
        Returns:
            Processed result
        """
        # For XY Plot generation, we need to create a grid of images
        # This would typically be handled in the worker function
        # Here we just return the result as-is
        return result
    
    # Using the centralized create_metadata method from BasePipeline
