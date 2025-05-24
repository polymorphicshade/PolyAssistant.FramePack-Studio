import torch
from .f1_generator import F1ModelGenerator

class F1WithEndframeModelGenerator(F1ModelGenerator):
    """
    Model generator for the F1 HunyuanVideo model with end frame support.
    This extends the F1 model with the ability to guide generation toward a specified end frame.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the F1 with Endframe model generator.
        """
        super().__init__(**kwargs)
        self.model_name = "F1 with Endframe"
        self.end_frame_applied = False
        # Inherits everything else from F1ModelGenerator
    
    def initialize_with_start_latent(self, history_latents, start_latent):
        """
        Initialize the history latents with the start latent for the F1 with Endframe model.
        This is the same as the F1 model, but we override it here for clarity.
        
        Args:
            history_latents: The history latents
            start_latent: The start latent
            
        Returns:
            The initialized history latents
        """
        # Add the start frame to history_latents
        return torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
    
    def update_history_latents(self, history_latents, generated_latents):
        """
        Update the history latents with the generated latents for the F1 with Endframe model.
        This overrides the F1 model's method to preserve the end frame latent.
        
        Args:
            history_latents: The history latents
            generated_latents: The generated latents
            
        Returns:
            The updated history latents
        """
        # If this is the first time we're updating history_latents after applying the end frame,
        # we need to preserve the end frame latent (which is at the last position)
        if not self.end_frame_applied:
            # Store the end frame latent
            self.end_frame_latent = history_latents[:, :, -1:, :, :].clone()
            self.end_frame_applied = True
        
        # Call the parent method to update history_latents
        updated_history_latents = super().update_history_latents(history_latents, generated_latents)
        
        # We don't apply the end frame latent here anymore - we'll only apply it in get_real_history_latents
        # when we're generating the final video
        
        return updated_history_latents
    
    def get_real_history_latents(self, history_latents, total_generated_latent_frames):
        """
        Get the real history latents for the F1 with Endframe model.
        This overrides the F1 model's method to ensure the end frame is included.
        
        Args:
            history_latents: The history latents
            total_generated_latent_frames: The total number of generated latent frames
            
        Returns:
            The real history latents
        """
        # Call the parent method to get the real history latents
        real_history_latents = super().get_real_history_latents(history_latents, total_generated_latent_frames)
        
        # If we have an end frame latent, make sure it's included in the real history latents
        if hasattr(self, 'end_frame_latent'):
            # For F1 model, the end frame should be the last frame
            real_history_latents[:, :, -1:, :, :] = self.end_frame_latent
        
        return real_history_latents
    
    def get_current_pixels(self, real_history_latents, section_latent_frames, vae):
        """
        Get the current pixels for the F1 with Endframe model.
        This overrides the F1 model's method to ensure the end frame is included.
        
        Args:
            real_history_latents: The real history latents
            section_latent_frames: The number of latent frames in the current section
            vae: The VAE model
            
        Returns:
            The current pixels
        """
        # Call the parent method to get the current pixels
        current_pixels = super().get_current_pixels(real_history_latents, section_latent_frames, vae)
        
        # We'll only apply the end frame in the final video output, not during intermediate sections
        # This is handled in get_real_history_latents
        
        return current_pixels
    
    def update_history_pixels(self, history_pixels, current_pixels, overlapped_frames):
        """
        Update the history pixels with the current pixels for the F1 with Endframe model.
        This overrides the F1 model's method to ensure the end frame is included.
        
        Args:
            history_pixels: The history pixels
            current_pixels: The current pixels
            overlapped_frames: The number of overlapped frames
            
        Returns:
            The updated history pixels
        """
        # Call the parent method to update history pixels
        updated_history_pixels = super().update_history_pixels(history_pixels, current_pixels, overlapped_frames)
        
        # We'll only apply the end frame in the final video output, not during intermediate sections
        # This is handled in get_real_history_latents
        
        return updated_history_pixels
