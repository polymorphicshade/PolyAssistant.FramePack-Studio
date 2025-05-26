"""
Video pipeline class for FramePack Studio.
This pipeline handles the "Video" model type.
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

class VideoPipeline(BasePipeline):
    """Pipeline for Video generation type."""
    
    def prepare_parameters(self, job_params):
        """
        Prepare parameters for the Video generation job.
        
        Args:
            job_params: Dictionary of job parameters
            
        Returns:
            Processed parameters dictionary
        """
        processed_params = job_params.copy()
        
        # Ensure we have the correct model type
        processed_params['model_type'] = "Video"
        
        return processed_params
    
    def validate_parameters(self, job_params):
        """
        Validate parameters for the Video generation job.
        
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
        
        # Check for input video path
        if not job_params.get('input_video_path'):
            return False, "Input video path is required for Video model"
        
        return True, None
    
    def preprocess_inputs(self, job_params):
        """
        Preprocess input video for the Video generation type.
        
        Args:
            job_params: Dictionary of job parameters
            
        Returns:
            Processed inputs dictionary
        """
        processed_inputs = {}
        
        # Get the input video path
        input_video_path = job_params.get('input_video_path')
        if not input_video_path:
            raise ValueError("Input video path is required for Video model")
        
        # Store the input video path
        processed_inputs['input_video'] = input_video_path
        
        # Get resolution parameters
        resolutionW = job_params.get('resolutionW', 640)
        resolutionH = job_params.get('resolutionH', 640)
        
        # Find nearest bucket size
        height, width = find_nearest_bucket(resolutionH, resolutionW, (resolutionW+resolutionH)/2)
        
        # Store the dimensions
        processed_inputs['height'] = height
        processed_inputs['width'] = width
        
        return processed_inputs
    
    def handle_results(self, job_params, result):
        """
        Handle the results of the Video generation.
        
        Args:
            job_params: The job parameters
            result: The generation result
            
        Returns:
            Processed result
        """
        # For Video generation, we just return the result as-is
        return result
    
    # Using the centralized create_metadata method from BasePipeline
