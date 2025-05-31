import os
import torch
import gc
import devicetorch
import warnings # Import the warnings module
import traceback # Ensure traceback is imported if used for logging

from pathlib import Path
from huggingface_hub import snapshot_download
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from .message_manager import MessageManager

# Get the directory of the current script (esrgan_core.py)
_MODULE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Constants
MODEL_ESRGAN_PATH = _MODULE_DIR / "model_esrgan"

class ESRGANUpscaler:
    def __init__(self, message_manager: MessageManager, device: torch.device):
        self.message_manager = message_manager
        self.device = device
        self.model_dir = Path(MODEL_ESRGAN_PATH)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.supported_models = {
            "RealESRGAN_x4plus": {
                "filename": "RealESRGAN_x4plus.pth",
                "hf_repo_id": "lllyasviel/Annotators", 
                "scale": 4,
                "model_class": RRDBNet, 
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            },
            "RealESRGAN_x2plus": { 
                "filename": "RealESRGAN_x2plus.pth",
                "hf_repo_id": "dtarnow/UPscaler", 
                "scale": 2,
                "model_class": RRDBNet,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            }
        }
        self.upsamplers = {} 

    def _ensure_model_downloaded(self, model_key: str) -> bool: 
        if model_key not in self.supported_models:
            self.message_manager.add_error(f"ESRGAN model key '{model_key}' not supported.")
            return False
            
        model_info = self.supported_models[model_key]
        model_filename = model_info["filename"]
        model_path = self.model_dir / model_filename

        if not model_path.exists():
            self.message_manager.add_message(f"ESRGAN model '{model_filename}' not found. Downloading...")
            try:
                snapshot_download(
                    repo_id=model_info["hf_repo_id"],
                    allow_patterns=[model_filename], 
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )
                if not model_path.exists():
                    self.message_manager.add_error(f"Failed to download '{model_filename}'. Please check the model source or place it manually in '{self.model_dir}'.")
                    return False
                self.message_manager.add_success(f"ESRGAN model '{model_filename}' downloaded successfully.")
            except Exception as e:
                self.message_manager.add_error(f"Failed to download ESRGAN model '{model_filename}': {e}")
                return False
        return True

    def load_model(self, target_scale: int) -> RealESRGANer | None:
        if target_scale in self.upsamplers and self.upsamplers[target_scale] is not None:
            self.message_manager.add_message(f"ESRGAN model for {target_scale}x already loaded.")
            return self.upsamplers[target_scale]

        model_key_to_load = None
        if target_scale == 4:
            model_key_to_load = "RealESRGAN_x4plus"
        elif target_scale == 2:
            model_key_to_load = "RealESRGAN_x2plus"
        else:
            self.message_manager.add_error(f"No pre-configured ESRGAN model for scale {target_scale}x.")
            return None 

        if not self._ensure_model_downloaded(model_key_to_load):
            return None

        model_info = self.supported_models[model_key_to_load]
        model_path_str = str(self.model_dir / model_info["filename"])
        
        self.message_manager.add_message(f"Loading ESRGAN model '{model_info['filename']}' for {target_scale}x scale to device: {self.device}...")
        try:
            model_arch = model_info["model_class"](**model_info["model_params"])
            
            gpu_id_for_realesrgan = None
            use_half_precision = False # Default to False
            if self.device.type == 'cuda':
                gpu_id_for_realesrgan = self.device.index if self.device.index is not None else 0
                use_half_precision = True # Enable half-precision for CUDA by default

            # Suppress the specific UserWarning from PyTorch related to weights_only
            with warnings.catch_warnings():
                # This filter targets the warning by category and a snippet of its message.
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=".*Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*"
                )
                
                upsampler = RealESRGANer(
                    scale=model_info["scale"],      
                    model_path=model_path_str,
                    dni_weight=None,      
                    model=model_arch, 
                    tile=0, # tile=0 means auto-tile, can be problematic with OOM. Consider setting a value e.g. 512.
                    tile_pad=10,
                    pre_pad=0,
                    half=use_half_precision, 
                    gpu_id=gpu_id_for_realesrgan 
                )
            self.upsamplers[target_scale] = upsampler
            self.message_manager.add_success(f"ESRGAN model '{model_info['filename']}' loaded for {target_scale}x scale.")
            return upsampler
        except Exception as e:
            self.message_manager.add_error(f"Failed to load ESRGAN model '{model_info['filename']}': {e}")
            self.message_manager.add_error(traceback.format_exc())
            if target_scale in self.upsamplers: del self.upsamplers[target_scale] 
            return None

    def unload_model(self, target_scale: int):
        if target_scale in self.upsamplers and self.upsamplers[target_scale] is not None:
            self.message_manager.add_message(f"Unloading ESRGAN model for {target_scale}x scale...")
            upsampler_instance = self.upsamplers.pop(target_scale)
            del upsampler_instance 
            devicetorch.empty_cache(torch) # Use your devicetorch helper
            gc.collect()
            self.message_manager.add_success(f"ESRGAN model for {target_scale}x unloaded and memory cleared.")
        else:
            self.message_manager.add_message(f"ESRGAN model for {target_scale}x not loaded, no need to unload.")

    def unload_all_models(self): # Renamed from unload_models for clarity
        if not self.upsamplers:
            self.message_manager.add_message("No ESRGAN models currently loaded.")
            return
            
        self.message_manager.add_message("Unloading all ESRGAN models...")
        scales_to_unload = list(self.upsamplers.keys()) 
        for scale in scales_to_unload:
            if scale in self.upsamplers: 
                upsampler_instance = self.upsamplers.pop(scale)
                del upsampler_instance
        devicetorch.empty_cache(torch) # Use your devicetorch helper
        gc.collect()
        self.message_manager.add_success("All ESRGAN models unloaded and memory cleared.")

    def upscale_frame(self, frame_np_array, target_scale: int):
        upsampler = self.upsamplers.get(target_scale)
        if upsampler is None:
            # Try to load it if not found, might be a good recovery mechanism
            self.message_manager.add_warning(f"ESRGAN model for {target_scale}x not loaded. Attempting to load now...")
            upsampler = self.load_model(target_scale)
            if upsampler is None:
                self.message_manager.add_error(f"Failed to auto-load ESRGAN model for {target_scale}x. Cannot upscale.")
                return None

        try:
            # RealESRGANer expects BGR, uint8 image. ImageIO often reads RGB.
            img_bgr = frame_np_array[:, :, ::-1] # Convert RGB to BGR
            output_bgr, _ = upsampler.enhance(img_bgr, outscale=target_scale) 
            output_rgb = output_bgr[:, :, ::-1] # Convert BGR back to RGB for consistency
            return output_rgb
        except Exception as e:
            self.message_manager.add_error(f"Error during ESRGAN frame upscaling: {e}")
            self.message_manager.add_error(traceback.format_exc()) # Keep traceback for this critical error
            if "out of memory" in str(e).lower() and self.device.type == 'cuda':
                self.message_manager.add_warning("CUDA OOM during upscaling. Emptying cache. Current model may need to be reloaded or consider smaller tile size.")
                devicetorch.empty_cache(torch)
            return None