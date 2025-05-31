import torch
import os
import numpy as np
import math
import decord
from tqdm import tqdm
import pathlib
from PIL import Image

from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import DynamicSwapInstaller
from diffusers_helper.utils import resize_and_center_crop
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.hunyuan import vae_encode, vae_decode
from .base_generator import BaseModelGenerator

class VideoModelGenerator(BaseModelGenerator):
    """
    Model generator for the Video extension of the Original HunyuanVideo model.
    This generator accepts video input instead of a single image.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Video model generator.
        """
        super().__init__(**kwargs)
        self.model_name = "Video"
        self.model_path = 'lllyasviel/FramePackI2V_HY'  # Same as Original
        self.model_repo_id_for_cache = "models--lllyasviel--FramePackI2V_HY"
        self.resolution = 640  # Default resolution
        self.no_resize = False  # Default to resize
        self.vae_batch_size = 16  # Default VAE batch size
        
        # Import decord and tqdm here to avoid import errors if not installed
        try:
            import decord
            from tqdm import tqdm
            self.decord = decord
            self.tqdm = tqdm
        except ImportError:
            print("Warning: decord or tqdm not installed. Video processing will not work.")
            self.decord = None
            self.tqdm = None
    
    def get_model_name(self):
        """
        Get the name of the model.
        """
        return self.model_name
    
    def load_model(self):
        """
        Load the Video transformer model.
        If offline mode is True, attempts to load from a local snapshot.
        """
        print(f"Loading {self.model_name} Transformer...")
        
        path_to_load = self.model_path # Initialize with the default path

        if self.offline:
            path_to_load = self._get_offline_load_path() # Calls the method in BaseModelGenerator
        
        # Create the transformer model
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            path_to_load, 
            torch_dtype=torch.bfloat16
        ).cpu()
        
        # Configure the model
        self.transformer.eval()
        self.transformer.to(dtype=torch.bfloat16)
        self.transformer.requires_grad_(False)
        
        # Set up dynamic swap if not in high VRAM mode
        if not self.high_vram:
            DynamicSwapInstaller.install_model(self.transformer, device=self.gpu)
        else:
            # In high VRAM mode, move the entire model to GPU
            self.transformer.to(device=self.gpu)
        
        print(f"{self.model_name} Transformer Loaded from {path_to_load}.")
        return self.transformer
    
    @torch.no_grad()
    def video_encode(self, video_path, resolution, no_resize=False, vae_batch_size=16, device=None, input_files_dir=None):
        """
        Encode a video into latent representations using the VAE.
        
        Args:
            video_path: Path to the input video file.
            resolution: Target resolution for resizing frames.
            no_resize: Whether to use the original video resolution.
            vae_batch_size: Number of frames to process per batch.
            device: Device for computation (e.g., "cuda").
            input_files_dir: Directory for input files that won't be cleaned up.
        
        Returns:
            A tuple containing:
            - start_latent: Latent of the first frame
            - input_image_np: First frame as numpy array
            - history_latents: Latents of all frames
            - fps: Frames per second of the input video
            - target_height: Target height of the video
            - target_width: Target width of the video
            - input_video_pixels: Video frames as tensor
            - end_of_input_video_image_np: Last frame as numpy array
        """
        if device is None:
            device = self.gpu
            
        # Normalize video path for Windows compatibility
        video_path = str(pathlib.Path(video_path).resolve())
        print(f"Processing video: {video_path}")
        
        # Check if the video is in the temp directory and if we have an input_files_dir
        if input_files_dir and "temp" in video_path:
            # Check if there's a copy of this video in the input_files_dir
            filename = os.path.basename(video_path)
            input_file_path = os.path.join(input_files_dir, filename)
            
            # If the file exists in input_files_dir, use that instead
            if os.path.exists(input_file_path):
                print(f"Using video from input_files_dir: {input_file_path}")
                video_path = input_file_path
            else:
                # If not, copy it to input_files_dir to prevent it from being deleted
                try:
                    from diffusers_helper.utils import generate_timestamp
                    safe_filename = f"{generate_timestamp()}_{filename}"
                    input_file_path = os.path.join(input_files_dir, safe_filename)
                    import shutil
                    shutil.copy2(video_path, input_file_path)
                    print(f"Copied video to input_files_dir: {input_file_path}")
                    video_path = input_file_path
                except Exception as e:
                    print(f"Error copying video to input_files_dir: {e}")

        # Check CUDA availability and fallback to CPU if needed
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU")
            device = "cpu"

        try:
            # Load video and get FPS
            print("Initializing VideoReader...")
            vr = decord.VideoReader(video_path)
            fps = vr.get_avg_fps()  # Get input video FPS
            num_real_frames = len(vr)
            print(f"Video loaded: {num_real_frames} frames, FPS: {fps}")

            # Truncate to nearest latent size (multiple of 4)
            latent_size_factor = 4
            num_frames = (num_real_frames // latent_size_factor) * latent_size_factor
            if num_frames != num_real_frames:
                print(f"Truncating video from {num_real_frames} to {num_frames} frames for latent size compatibility")
            num_real_frames = num_frames

            # Read frames
            print("Reading video frames...")
            frames = vr.get_batch(range(num_real_frames)).asnumpy()  # Shape: (num_real_frames, height, width, channels)
            print(f"Frames read: {frames.shape}")

            # Get native video resolution
            native_height, native_width = frames.shape[1], frames.shape[2]
            print(f"Native video resolution: {native_width}x{native_height}")
        
            # Use native resolution if height/width not specified, otherwise use provided values
            target_height = native_height
            target_width = native_width
        
            # Adjust to nearest bucket for model compatibility
            if not no_resize:
                target_height, target_width = find_nearest_bucket(target_height, target_width, resolution=resolution)
                print(f"Adjusted resolution: {target_width}x{target_height}")
            else:
                print(f"Using native resolution without resizing: {target_width}x{target_height}")

            # Preprocess frames to match original image processing
            processed_frames = []
            for i, frame in enumerate(frames):
                frame_np = resize_and_center_crop(frame, target_width=target_width, target_height=target_height)
                processed_frames.append(frame_np)
            processed_frames = np.stack(processed_frames)  # Shape: (num_real_frames, height, width, channels)
            print(f"Frames preprocessed: {processed_frames.shape}")

            # Save first frame for CLIP vision encoding
            input_image_np = processed_frames[0]
            end_of_input_video_image_np = processed_frames[-1]

            # Convert to tensor and normalize to [-1, 1]
            print("Converting frames to tensor...")
            frames_pt = torch.from_numpy(processed_frames).float() / 127.5 - 1
            frames_pt = frames_pt.permute(0, 3, 1, 2)  # Shape: (num_real_frames, channels, height, width)
            frames_pt = frames_pt.unsqueeze(0)  # Shape: (1, num_real_frames, channels, height, width)
            frames_pt = frames_pt.permute(0, 2, 1, 3, 4)  # Shape: (1, channels, num_real_frames, height, width)
            print(f"Tensor shape: {frames_pt.shape}")
            
            # Save pixel frames for use in worker
            input_video_pixels = frames_pt.cpu()

            # Move to device
            print(f"Moving tensor to device: {device}")
            frames_pt = frames_pt.to(device)
            print("Tensor moved to device")

            # Move VAE to device
            print(f"Moving VAE to device: {device}")
            self.vae.to(device)
            print("VAE moved to device")

            # Encode frames in batches
            print(f"Encoding input video frames in VAE batch size {vae_batch_size}")
            latents = []
            self.vae.eval()
            with torch.no_grad():
                for i in tqdm(range(0, frames_pt.shape[2], vae_batch_size), desc="Encoding video frames", mininterval=0.1):
                    batch = frames_pt[:, :, i:i + vae_batch_size]  # Shape: (1, channels, batch_size, height, width)
                    try:
                        # Log GPU memory before encoding
                        if device == "cuda":
                            free_mem = torch.cuda.memory_allocated() / 1024**3
                        batch_latent = vae_encode(batch, self.vae)
                        # Synchronize CUDA to catch issues
                        if device == "cuda":
                            torch.cuda.synchronize()
                        latents.append(batch_latent)
                    except RuntimeError as e:
                        print(f"Error during VAE encoding: {str(e)}")
                        if device == "cuda" and "out of memory" in str(e).lower():
                            print("CUDA out of memory, try reducing vae_batch_size or using CPU")
                        raise
            
            # Concatenate latents
            print("Concatenating latents...")
            history_latents = torch.cat(latents, dim=2)  # Shape: (1, channels, frames, height//8, width//8)
            print(f"History latents shape: {history_latents.shape}")

            # Get first frame's latent
            start_latent = history_latents[:, :, :1]  # Shape: (1, channels, 1, height//8, width//8)
            print(f"Start latent shape: {start_latent.shape}")

            # Move VAE back to CPU to free GPU memory
            if device == "cuda":
                self.vae.to(self.cpu)
                torch.cuda.empty_cache()
                print("VAE moved back to CPU, CUDA cache cleared")

            return start_latent, input_image_np, history_latents, fps, target_height, target_width, input_video_pixels, end_of_input_video_image_np

        except Exception as e:
            print(f"Error in video_encode: {str(e)}")
            raise
    
    def prepare_history_latents(self, height, width):
        """
        Prepare the history latents tensor for the Video model.
        
        Args:
            height: The height of the image
            width: The width of the image
            
        Returns:
            The initialized history latents tensor
        """
        return torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8), 
            dtype=torch.float32
        ).cpu()
    
    def get_latent_paddings(self, total_latent_sections):
        """
        Get the latent paddings for the Video model.
        
        Args:
            total_latent_sections: The total number of latent sections
            
        Returns:
            A list of latent paddings
        """
        # Video model uses reversed latent paddings like Original
        if total_latent_sections > 4:
            return [3] + [2] * (total_latent_sections - 3) + [1, 0]
        else:
            return list(reversed(range(total_latent_sections)))

    def prepare_clean_latents_and_indices(self, end_latent, end_frame_weight, end_clip_embedding, end_of_input_video_embedding, latent_paddings, latent_padding, latent_padding_size, latent_window_size, video_latents, history_latents):
        """
        Combined method to prepare clean latents and indices for the Video model.
        
        Args:
            Work in progress - better not to pass in latent_paddings and latent_padding.
            
        Returns:
            A tuple of (clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices, clean_latents, clean_latents_2x, clean_latents_4x)
        """
        # HACK SOME STUFF IN THAT SHOULD NOT BE HERE
        # num_clean_frames Should come from UI - 20250506 pftq: Renamed slider to Number of Context Frames and updated description
        # num_clean_frames = gr.Slider(label="Number of Context Frames (Adherence to Video)", minimum=2, maximum=10, value=5, step=1, info="Expensive. Retain more video details. Reduce if memory issues or motion too restricted (jumpcut, ignoring prompt, still).")
        num_clean_frames = 5 # This is a placeholder for the number of clean frames. SEE: 20250507 pftq: Process end frame if provided
        # Placeholders for end frame processing
        # Colin, I'm only leaving them for the moment in case you want separate models for
        # Video-backward and Video-backward-Endframe.
        # end_latent = None
        # end_of_input_video_embedding = None # Placeholder for end frame's CLIP embedding. SEE: 20250507 pftq: Process end frame if provided
        # end_clip_embedding = None # Placeholders for end frame processing. SEE: 20250507 pftq: Process end frame if provided
        # end_frame_weight = 0.0 # Placeholders for end frame processing. SEE: 20250507 pftq: Process end frame if provided
        # HACK MORE STUFF IN THAT PROBABLY SHOULD BE ARGUMENTS OR OTHWISE MADE AVAILABLE
        end_of_input_video_latent = video_latents[:, :, -1:] # Last frame of the input video (produced by video_encode in the PR)
        is_start_of_video = latent_padding == 0 # This refers to the start of the *generated* video part
        is_end_of_video = latent_padding == latent_paddings[0] # This refers to the end of the *generated* video part (closest to input video) (better not to pass in latent_paddings[])
        # End of HACK STUFF

        # Dynamic frame allocation for context frames (clean latents)
        # This determines which frames from history_latents are used as input for the transformer.
        available_frames = video_latents.shape[2] if is_start_of_video else history_latents.shape[2] # Use input video frames for first segment, else previously generated history
        effective_clean_frames = max(0, num_clean_frames - 1) if num_clean_frames > 1 else 1
        if is_start_of_video:
            effective_clean_frames = 1 # Avoid jumpcuts if input video is too different

        clean_latent_pre_frames = effective_clean_frames 
        num_2x_frames = min(2, max(1, available_frames - clean_latent_pre_frames - 1)) if available_frames > clean_latent_pre_frames + 1 else 1
        num_4x_frames = min(16, max(1, available_frames - clean_latent_pre_frames - num_2x_frames)) if available_frames > clean_latent_pre_frames + num_2x_frames else 1
        total_context_frames = num_2x_frames + num_4x_frames
        total_context_frames = min(total_context_frames, available_frames - clean_latent_pre_frames)

        # Prepare indices for the transformer's input (these define the *relative positions* of different frame types in the input tensor)
        # The total length is the sum of various frame types:
        # clean_latent_pre_frames: frames before the blank/generated section
        # latent_padding_size: blank frames before the generated section (for backward generation)
        # latent_window_size: the new frames to be generated
        # post_frames: frames after the generated section
        # num_2x_frames, num_4x_frames: frames for lower resolution context
        # 20250511 pftq: Dynamically adjust post_frames based on clean_latents_post
        post_frames = 1 if is_end_of_video and end_latent is not None else effective_clean_frames  # 20250511 pftq: Single frame for end_latent, otherwise padding causes still image
        indices = torch.arange(0, clean_latent_pre_frames + latent_padding_size + latent_window_size + post_frames + num_2x_frames + num_4x_frames).unsqueeze(0)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(
            [clean_latent_pre_frames, latent_padding_size, latent_window_size, post_frames, num_2x_frames, num_4x_frames], dim=1
        )
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1) # Combined indices for 1x clean latents

        # Prepare the *actual latent data* for the transformer's context inputs
        # These are extracted from history_latents (or video_latents for the first segment)
        context_frames = history_latents[:, :, -(total_context_frames + clean_latent_pre_frames):-clean_latent_pre_frames, :, :] if total_context_frames > 0 else history_latents[:, :, :1, :, :]
        # clean_latents_4x: 4x downsampled context frames. From history_latents (or video_latents).
        # clean_latents_2x: 2x downsampled context frames. From history_latents (or video_latents).
        split_sizes = [num_4x_frames, num_2x_frames]
        split_sizes = [s for s in split_sizes if s > 0]
        if split_sizes and context_frames.shape[2] >= sum(split_sizes):
            splits = context_frames.split(split_sizes, dim=2)
            split_idx = 0
            clean_latents_4x = splits[split_idx] if num_4x_frames > 0 else history_latents[:, :, :1, :, :]
            split_idx += 1 if num_4x_frames > 0 else 0
            clean_latents_2x = splits[split_idx] if num_2x_frames > 0 and split_idx < len(splits) else history_latents[:, :, :1, :, :]
        else:
            clean_latents_4x = clean_latents_2x = history_latents[:, :, :1, :, :]

        # clean_latents_pre: Latents from the *end* of the input video (if is_start_of_video), or previously generated history.
        # Its purpose is to provide a smooth transition *from* the input video.
        clean_latents_pre = video_latents[:, :, -min(effective_clean_frames, video_latents.shape[2]):].to(history_latents)
        
        # clean_latents_post: Latents from the *beginning* of the previously generated video segments.
        # Its purpose is to provide a smooth transition *to* the existing generated content.
        clean_latents_post = history_latents[:, :, :min(effective_clean_frames, history_latents.shape[2]), :, :]

        # Special handling for the end frame:
        # If it's the very first segment being generated (is_end_of_video in terms of generation order),
        # and an end_latent was provided, force clean_latents_post to be that end_latent.
        if is_end_of_video:
            clean_latents_post = torch.zeros_like(end_of_input_video_latent).to(history_latents) # Initialize to zero
        
        if end_latent is not None:
            # image_encoder_last_hidden_state: Weighted average of CLIP embedding of first input frame and end frame's CLIP embedding
            # This guides the overall content to transition towards the end frame.
            image_encoder_last_hidden_state = (1 - end_frame_weight) * end_of_input_video_embedding + end_clip_embedding * end_frame_weight
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(self.transformer.dtype)
            
            if is_end_of_video:
                # For the very first generated segment, the "post" part is the end_latent itself.
                clean_latents_post = end_latent.to(history_latents)[:, :, :1, :, :] # Ensure single frame
            
        # Pad clean_latents_pre/post if they have fewer frames than specified by clean_latent_pre_frames/post_frames
        if clean_latents_pre.shape[2] < clean_latent_pre_frames:
            clean_latents_pre = clean_latents_pre.repeat(1, 1, math.ceil(clean_latent_pre_frames / clean_latents_pre.shape[2]), 1, 1)[:,:,:clean_latent_pre_frames]
        if clean_latents_post.shape[2] < post_frames:
            clean_latents_post = clean_latents_post.repeat(1, 1, math.ceil(post_frames / clean_latents_post.shape[2]), 1, 1)[:,:,:post_frames]
            
        # clean_latents: Concatenation of pre and post clean latents. These are the 1x resolution context frames.
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

        return clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices, clean_latents, clean_latents_2x, clean_latents_4x
    
    def prepare_indices(self, latent_padding_size, latent_window_size):
        """
        Prepare the indices for the Video model.
        
        Args:
            latent_padding_size: The size of the latent padding
            latent_window_size: The size of the latent window
            
        Returns:
            A tuple of (clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices)
        """
        indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
        
        return clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices
    
    # Store the full video latents for context
    full_video_latents = None
    
    def set_full_video_latents(self, video_latents):
        """
        Set the full video latents for context.
        
        Args:
            video_latents: The full video latents
        """
        self.full_video_latents = video_latents
    
    def prepare_clean_latents(self, start_latent, history_latents):
        """
        Prepare the clean latents for the Video model.
        
        Args:
            start_latent: The start latent
            history_latents: The history latents
            
        Returns:
            A tuple of (clean_latents, clean_latents_2x, clean_latents_4x)
        """
        # Get the actual size of the history_latents tensor
        history_frames = history_latents.shape[2]
        
        # Prepare the clean latents
        clean_latents_pre = start_latent.to(history_latents)
        
        # For clean_latents_post, use the last frame from history_latents if available
        if history_frames > 0:
            clean_latents_post = history_latents[:, :, -1:, :, :]
        else:
            clean_latents_post = torch.zeros_like(clean_latents_pre)
        
        # For clean_latents_2x, prioritize using frames from history_latents for better coherence
        # between sections, but fall back to full_video_latents if needed
        if history_frames > 1:
            # Use frames from history_latents for better coherence between sections
            frames_2x = min(2, history_frames)
            clean_latents_2x = history_latents[:, :, -frames_2x:, :, :].to(history_latents)
            print(f"Using {frames_2x} frames from history_latents for clean_latents_2x")
        elif self.full_video_latents is not None and self.full_video_latents.shape[2] > 1:
            # Fall back to full_video_latents if history_latents doesn't have enough frames
            video_frames = self.full_video_latents.shape[2]
            if video_frames >= 2:
                clean_latents_2x = self.full_video_latents[:, :, -2:, :, :].to(history_latents)
                print(f"Using 2 frames from full_video_latents for clean_latents_2x")
            else:
                clean_latents_2x = torch.zeros((1, 16, 2, history_latents.shape[3], history_latents.shape[4]), dtype=history_latents.dtype, device=history_latents.device)
                if video_frames == 1:
                    clean_latents_2x[:, :, 0:1, :, :] = self.full_video_latents
                print(f"Using {video_frames} frames from full_video_latents for clean_latents_2x")
        else:
            # If neither history_latents nor full_video_latents have enough frames, use zeros
            clean_latents_2x = torch.zeros((1, 16, 2, history_latents.shape[3], history_latents.shape[4]), dtype=history_latents.dtype, device=history_latents.device)
            print("Using zeros for clean_latents_2x")
        
        # For clean_latents_4x, combine frames from history_latents and full_video_latents
        # to provide better context for coherence
        if history_frames > 0:
            # Determine how many frames to use from history_latents
            history_frames_4x = min(8, history_frames)  # Use up to 8 frames from history_latents
            history_frames_part = history_latents[:, :, -history_frames_4x:, :, :].to(history_latents)
            print(f"Using {history_frames_4x} frames from history_latents for clean_latents_4x")
            
            # Determine how many frames to use from full_video_latents
            remaining_frames = 16 - history_frames_4x
            if remaining_frames > 0 and self.full_video_latents is not None and self.full_video_latents.shape[2] > 0:
                video_frames = min(remaining_frames, self.full_video_latents.shape[2])
                video_frames_part = self.full_video_latents[:, :, -video_frames:, :, :].to(history_latents)
                print(f"Using {video_frames} frames from full_video_latents for clean_latents_4x")
                
                # If we still need more frames, pad with zeros
                if history_frames_4x + video_frames < 16:
                    padding_size = 16 - (history_frames_4x + video_frames)
                    padding = torch.zeros((1, 16, padding_size, history_latents.shape[3], history_latents.shape[4]), dtype=history_latents.dtype, device=history_latents.device)
                    clean_latents_4x = torch.cat([padding, video_frames_part, history_frames_part], dim=2)
                    print(f"Added {padding_size} padding frames for clean_latents_4x")
                else:
                    clean_latents_4x = torch.cat([video_frames_part, history_frames_part], dim=2)
            else:
                # If we don't have full_video_latents, pad with zeros
                padding_size = 16 - history_frames_4x
                if padding_size > 0:
                    padding = torch.zeros((1, 16, padding_size, history_latents.shape[3], history_latents.shape[4]), dtype=history_latents.dtype, device=history_latents.device)
                    clean_latents_4x = torch.cat([padding, history_frames_part], dim=2)
                    print(f"Added {padding_size} padding frames for clean_latents_4x (no video frames)")
                else:
                    clean_latents_4x = history_frames_part
        elif self.full_video_latents is not None and self.full_video_latents.shape[2] > 0:
            # If no history frames, use frames from full_video_latents
            video_frames = min(16, self.full_video_latents.shape[2])
            if video_frames > 0:
                clean_latents_4x = self.full_video_latents[:, :, -video_frames:, :, :].to(history_latents)
                # If we have fewer than 16 frames, pad with zeros
                if video_frames < 16:
                    padding = torch.zeros((1, 16, 16 - video_frames, history_latents.shape[3], history_latents.shape[4]), dtype=history_latents.dtype, device=history_latents.device)
                    clean_latents_4x = torch.cat([padding, clean_latents_4x], dim=2)
                    print(f"Using {video_frames} frames from full_video_latents with {16 - video_frames} padding frames for clean_latents_4x")
                else:
                    print(f"Using {video_frames} frames from full_video_latents for clean_latents_4x")
            else:
                clean_latents_4x = torch.zeros((1, 16, 16, history_latents.shape[3], history_latents.shape[4]), dtype=history_latents.dtype, device=history_latents.device)
                print("Using zeros for clean_latents_4x (no video frames)")
        else:
            # If neither history_latents nor full_video_latents have frames, use zeros
            clean_latents_4x = torch.zeros((1, 16, 16, history_latents.shape[3], history_latents.shape[4]), dtype=history_latents.dtype, device=history_latents.device)
            print("Using zeros for clean_latents_4x (no history or video frames)")
        
        # Concatenate the pre and post latents
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        
        return clean_latents, clean_latents_2x, clean_latents_4x
    
    def update_history_latents(self, history_latents, generated_latents):
        """
        Update the history latents with the generated latents for the Video model.
        
        Args:
            history_latents: The history latents
            generated_latents: The generated latents
            
        Returns:
            The updated history latents
        """
        # For Video model, we prepend the generated latents to the history latents
        # This matches the original implementation in video-example.py
        # It generates new sections backwards in time, chunk by chunk
        return torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
    
    def get_real_history_latents(self, history_latents, total_generated_latent_frames):
        """
        Get the real history latents for the Video model.
        
        Args:
            history_latents: The history latents
            total_generated_latent_frames: The total number of generated latent frames
            
        Returns:
            The real history latents
        """
        return history_latents[:, :, :total_generated_latent_frames, :, :]
    
    def update_history_pixels(self, history_pixels, current_pixels, overlapped_frames):
        """
        Update the history pixels with the current pixels for the Video model.
        
        Args:
            history_pixels: The history pixels
            current_pixels: The current pixels
            overlapped_frames: The number of overlapped frames
            
        Returns:
            The updated history pixels
        """
        from diffusers_helper.utils import soft_append_bcthw
        # For Video model, we prepend the current pixels to the history pixels
        # This matches the original implementation in video-example.py
        return soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
    
    def get_section_latent_frames(self, latent_window_size, is_last_section):
        """
        Get the number of section latent frames for the Video model.
        
        Args:
            latent_window_size: The size of the latent window
            is_last_section: Whether this is the last section
            
        Returns:
            The number of section latent frames
        """
        return latent_window_size * 2
    
    def get_current_pixels(self, real_history_latents, section_latent_frames, vae):
        """
        Get the current pixels for the Video model.
        
        Args:
            real_history_latents: The real history latents
            section_latent_frames: The number of section latent frames
            vae: The VAE model
            
        Returns:
            The current pixels
        """
        return vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
    
    def format_position_description(self, total_generated_latent_frames, current_pos, original_pos, current_prompt):
        """
        Format the position description for the Video model.
        
        Args:
            total_generated_latent_frames: The total number of generated latent frames
            current_pos: The current position in seconds (includes input video time)
            original_pos: The original position in seconds
            current_prompt: The current prompt
            
        Returns:
            The formatted position description
        """
        # For Video model, current_pos already includes the input video time
        # We just need to display the total generated frames and the current position
        return (f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, '
                f'Generated video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). '
                f'Current position: {current_pos:.2f}s (remaining: {original_pos:.2f}s). '
                f'using prompt: {current_prompt[:256]}...')
    
    def combine_videos(self, source_video_path, generated_video_path, output_path):
        """
        Combine the source video with the generated video side by side.
        
        Args:
            source_video_path: Path to the source video
            generated_video_path: Path to the generated video
            output_path: Path to save the combined video
            
        Returns:
            Path to the combined video
        """
        try:
            import os
            import subprocess
            
            print(f"Combining source video {source_video_path} with generated video {generated_video_path}")
            
            # Get the ffmpeg executable from the VideoProcessor class
            from modules.toolbox.toolbox_processor import VideoProcessor
            from modules.toolbox.message_manager import MessageManager
            
            # Create a message manager for logging
            message_manager = MessageManager()
            
            # Import settings from main module
            try:
                from __main__ import settings
                video_processor = VideoProcessor(message_manager, settings.settings)
            except ImportError:
                # Fallback to creating a new settings object
                from modules.settings import Settings
                settings = Settings()
                video_processor = VideoProcessor(message_manager, settings.settings)
            
            # Get the ffmpeg executable
            ffmpeg_exe = video_processor.ffmpeg_exe
            
            if not ffmpeg_exe:
                print("FFmpeg executable not found. Cannot combine videos.")
                return None
            
            print(f"Using ffmpeg at: {ffmpeg_exe}")
            
            # Create a temporary directory for the filter script
            import tempfile
            temp_dir = tempfile.gettempdir()
            filter_script_path = os.path.join(temp_dir, f"filter_script_{os.path.basename(output_path)}.txt")
            
            # Get video dimensions using ffprobe
            def get_video_info(video_path):
                cmd = [
                    ffmpeg_exe, "-i", video_path, 
                    "-hide_banner", "-loglevel", "error"
                ]
                
                # Run ffmpeg to get video info (it will fail but output info to stderr)
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Parse the output to get dimensions
                width = height = None
                for line in result.stderr.split('\n'):
                    if 'Video:' in line:
                        # Look for dimensions like 640x480
                        import re
                        match = re.search(r'(\d+)x(\d+)', line)
                        if match:
                            width = int(match.group(1))
                            height = int(match.group(2))
                            break
                
                return width, height
            
            # Get dimensions of both videos
            source_width, source_height = get_video_info(source_video_path)
            generated_width, generated_height = get_video_info(generated_video_path)
            
            if not source_width or not generated_width:
                print("Error: Could not determine video dimensions")
                return None
            
            print(f"Source video: {source_width}x{source_height}")
            print(f"Generated video: {generated_width}x{generated_height}")
            
            # Calculate target dimensions (maintain aspect ratio)
            target_height = max(source_height, generated_height)
            source_target_width = int(source_width * (target_height / source_height))
            generated_target_width = int(generated_width * (target_height / generated_height))
            
            # Create a complex filter for side-by-side display with labels
            filter_complex = (
                f"[0:v]scale={source_target_width}:{target_height}[left];"
                f"[1:v]scale={generated_target_width}:{target_height}[right];"
                f"[left]drawtext=text='Source':x=({source_target_width}/2-50):y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[left_text];"
                f"[right]drawtext=text='Generated':x=({generated_target_width}/2-70):y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[right_text];"
                f"[left_text][right_text]hstack=inputs=2[v]"
            )
            
            # Write the filter script to a file
            with open(filter_script_path, 'w') as f:
                f.write(filter_complex)
            
            # Build the ffmpeg command
            cmd = [
                ffmpeg_exe, "-y",
                "-i", source_video_path,
                "-i", generated_video_path,
                "-filter_complex_script", filter_script_path,
                "-map", "[v]"
            ]
            
            # Check if source video has audio
            has_audio_cmd = [
                ffmpeg_exe, "-i", source_video_path,
                "-hide_banner", "-loglevel", "error"
            ]
            audio_check = subprocess.run(has_audio_cmd, capture_output=True, text=True)
            has_audio = "Audio:" in audio_check.stderr
            
            if has_audio:
                cmd.extend(["-map", "0:a"])
            
            # Add output options
            cmd.extend([
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium"
            ])
            
            if has_audio:
                cmd.extend(["-c:a", "aac"])
            
            cmd.append(output_path)
            
            # Run the ffmpeg command
            print(f"Running ffmpeg command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Clean up the filter script
            if os.path.exists(filter_script_path):
                os.remove(filter_script_path)
            
            print(f"Combined video saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error combining videos: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
