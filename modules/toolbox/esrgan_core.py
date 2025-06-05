import os
import torch
import gc
import devicetorch
import warnings 
import traceback 

from pathlib import Path
from huggingface_hub import snapshot_download
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url # Import for direct downloads

# Conditional import for GFPGAN
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

from .message_manager import MessageManager

_MODULE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_ESRGAN_PATH = _MODULE_DIR / "model_esrgan"
# Define a path for GFPGAN models, can be within MODEL_ESRGAN_PATH or separate
MODEL_GFPGAN_PATH = _MODULE_DIR / "model_gfpgan" 

class ESRGANUpscaler:
    def __init__(self, message_manager: MessageManager, device: torch.device):
        self.message_manager = message_manager
        self.device = device
        self.model_dir = Path(MODEL_ESRGAN_PATH)
        self.gfpgan_model_dir = Path(MODEL_GFPGAN_PATH) # GFPGAN model directory
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.gfpgan_model_dir, exist_ok=True) # Ensure GFPGAN model dir exists
        
        self.supported_models = {
            "RealESRGAN_x4plus": {
                "filename": "RealESRGAN_x4plus.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", # Official URL
                "hf_repo_id": None, # Can be a fallback if file_url fails or for other models
                "scale": 4,
                "model_class": RRDBNet, 
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                "description": "General purpose x4 upscaling model. Good choice for most videos."
            },
            "RealESRGAN_x2plus": { 
                "filename": "RealESRGAN_x2plus.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", # Official URL
                "hf_repo_id": None,
                "scale": 2,
                "model_class": RRDBNet,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
                "description": "General purpose x2 model. Faster with a smaller output size than its big brother."
            },
            "RealESRGAN_x4plus_anime_6B": {
                "filename": "RealESRGAN_x4plus_anime_6B.pth",
                # Using ximso's HF repo as the official PTH on GitHub is part of a zip for older releases.
                # If a direct PTH link for 6B becomes standard on official releases, it can be added.
                # "hf_repo_id": "ximso/RealESRGAN_x4plus_anime_6B", # Removed as file_url is now primary
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth", # Official URL
                "hf_repo_id": None, # Set to None as file_url is primary. Can add HF as fallback if needed.
                "scale": 4,
                "model_class": RRDBNet, 
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
                "description": "Optimized for x4 anime-style video (lighter 6-block version)."
            },
            "RealESR AnimeVideo v3 (x4)": {
                "filename": "realesr-animevideov3.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth", # Official URL
                "hf_repo_id": None,
                "scale": 4,
                "model_class": SRVGGNetCompact,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'),
                "description": "Specialized x4 model for anime video (SRVGG based)."
            }
        }
        
        self.upsamplers: dict[str, dict[str, RealESRGANer | int | None]] = {} 
        self.face_enhancer: GFPGANer | None = None # For GFPGAN

    def _ensure_model_downloaded(self, model_key: str, target_dir: Path | None = None, is_gfpgan: bool = False) -> bool:
        # Modified to handle GFPGAN model download as well
        # If target_dir is not provided, use default based on is_gfpgan
        if target_dir is None:
            current_model_dir = self.gfpgan_model_dir if is_gfpgan else self.model_dir
        else:
            current_model_dir = target_dir

        if is_gfpgan: # Special handling for GFPGAN model info
            model_info = { # Hardcode GFPGAN info for now, can be made more flexible
                "filename": "GFPGANv1.4.pth", # Or GFPGANv1.3.pth
                "file_url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
                # "file_url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                "hf_repo_id": None 
            }
            model_filename = model_info["filename"]
        else: # ESRGAN models
            if model_key not in self.supported_models:
                self.message_manager.add_error(f"ESRGAN model key '{model_key}' not supported.")
                return False
            model_info = self.supported_models[model_key]
            model_filename = model_info["filename"]

        model_path = current_model_dir / model_filename

        if not model_path.exists():
            self.message_manager.add_message(f"Model '{model_filename}' not found. Downloading...")
            try:
                downloaded_successfully = False
                # Prioritize file_url if available
                if "file_url" in model_info and model_info["file_url"]:
                    urls_to_try = model_info["file_url"]
                    if isinstance(urls_to_try, str):
                        urls_to_try = [urls_to_try]
                    
                    for url in urls_to_try:
                        self.message_manager.add_message(f"Attempting download from URL: {url}")
                        try:
                            load_file_from_url(
                                url=url, 
                                model_dir=str(current_model_dir),
                                progress=True, 
                                file_name=model_filename # Ensure correct filename
                            )
                            if model_path.exists():
                                downloaded_successfully = True
                                self.message_manager.add_success(f"Model '{model_filename}' downloaded from URL.")
                                break 
                        except Exception as e_url:
                            self.message_manager.add_warning(f"Failed to download from {url}: {e_url}. Trying next source if available.")
                            continue
                
                # Fallback to Hugging Face Hub if file_url failed or not provided, and hf_repo_id is set
                if not downloaded_successfully and "hf_repo_id" in model_info and model_info["hf_repo_id"]:
                    self.message_manager.add_message(f"Attempting download from Hugging Face Hub: {model_info['hf_repo_id']}")
                    snapshot_download(
                        repo_id=model_info["hf_repo_id"],
                        allow_patterns=[model_filename], 
                        local_dir=current_model_dir, # Use current_model_dir
                        local_dir_use_symlinks=False
                    )
                    if model_path.exists():
                        downloaded_successfully = True
                        self.message_manager.add_success(f"Model '{model_filename}' downloaded from Hugging Face Hub.")

                if not downloaded_successfully:
                    self.message_manager.add_error(f"All download attempts failed for '{model_filename}'.")
                    return False

            except Exception as e:
                self.message_manager.add_error(f"Failed to download model '{model_filename}': {e}")
                self.message_manager.add_error(traceback.format_exc())
                return False
        return True

    def load_model(self, model_key: str, tile_size: int = 0) -> RealESRGANer | None:
        if model_key not in self.supported_models:
            self.message_manager.add_error(f"ESRGAN model key '{model_key}' not supported.")
            return None

        if model_key in self.upsamplers:
            existing_config = self.upsamplers[model_key]
            log_tile_size = str(tile_size) if tile_size > 0 else "Auto"
            existing_log_tile_size = str(existing_config.get('tile_size', 0)) if existing_config.get('tile_size', 0) > 0 else "Auto"

            if existing_config and existing_config.get("upsampler") is not None and existing_config.get("tile_size") == tile_size:
                self.message_manager.add_message(f"ESRGAN model '{model_key}' with tile size {log_tile_size} already loaded.")
                return existing_config["upsampler"] # type: ignore
            elif existing_config and existing_config.get("upsampler") is not None and existing_config.get("tile_size") != tile_size:
                self.message_manager.add_message(
                    f"ESRGAN model '{model_key}' loaded with tile size {existing_log_tile_size}. "
                    f"Unloading to reload with new tile size {log_tile_size}."
                )
                self.unload_model(model_key) 

        if not self._ensure_model_downloaded(model_key):
            return None

        model_info = self.supported_models[model_key]
        model_path_str = str(self.model_dir / model_info["filename"])
        log_tile_size = str(tile_size) if tile_size > 0 else "Auto"
        
        self.message_manager.add_message(f"Loading ESRGAN model '{model_info['filename']}' (Key: {model_key}, Scale: {model_info['scale']}x, Tile: {log_tile_size}) to device: {self.device}...")
        try:
            model_params_with_correct_scale = model_info["model_params"].copy()
            # Ensure 'scale' or 'upscale' matches model_info['scale'] for clarity, though RealESRGANer uses its own scale param
            if "scale" in model_params_with_correct_scale:
                 model_params_with_correct_scale["scale"] = model_info["scale"]
            elif "upscale" in model_params_with_correct_scale: # SRVGGNetCompact uses 'upscale'
                 model_params_with_correct_scale["upscale"] = model_info["scale"]
            else: # Add 'scale' if missing (e.g. if model_params was very minimal)
                 model_params_with_correct_scale["scale"] = model_info["scale"]


            model_arch = model_info["model_class"](**model_params_with_correct_scale)
            
            gpu_id_for_realesrgan = None
            use_half_precision = False 
            if self.device.type == 'cuda':
                gpu_id_for_realesrgan = self.device.index if self.device.index is not None else 0
                use_half_precision = True 

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=".*Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*"
                )
                upsampler = RealESRGANer(
                    scale=model_info["scale"], # This is the crucial scale for RealESRGANer
                    model_path=model_path_str,
                    dni_weight=None,      
                    model=model_arch, 
                    tile=tile_size,
                    tile_pad=10,
                    pre_pad=0,
                    half=use_half_precision, 
                    gpu_id=gpu_id_for_realesrgan 
                )
            self.upsamplers[model_key] = {"upsampler": upsampler, "tile_size": tile_size, "native_scale": model_info["scale"]}
            self.message_manager.add_success(f"ESRGAN model '{model_info['filename']}' (Key: {model_key}, Scale: {model_info['scale']}x, Tile: {log_tile_size}) loaded.")
            return upsampler
        except Exception as e:
            self.message_manager.add_error(f"Failed to load ESRGAN model '{model_info['filename']}' (Key: {model_key}): {e}")
            self.message_manager.add_error(traceback.format_exc())
            if model_key in self.upsamplers: del self.upsamplers[model_key] 
            return None

    def _load_face_enhancer(self, model_name="GFPGANv1.4.pth", bg_upsampler=None) -> bool:
        if not GFPGAN_AVAILABLE:
            self.message_manager.add_warning("GFPGAN library not available. Cannot load face enhancer.")
            return False
        if self.face_enhancer is not None:
            # If bg_upsampler changed, we might need to re-init. For now, assume if loaded, it's fine or will be handled by caller.
            if bg_upsampler is not None and hasattr(self.face_enhancer, 'bg_upsampler') and self.face_enhancer.bg_upsampler != bg_upsampler:
                self.message_manager.add_message("GFPGAN face enhancer already loaded, but with a different background upsampler. Re-initializing GFPGAN...")
                self._unload_face_enhancer() # Unload to reload with new bg_upsampler
            else:
                self.message_manager.add_message("GFPGAN face enhancer already loaded.")
                return True


        if not self._ensure_model_downloaded(model_key=model_name, is_gfpgan=True): 
            self.message_manager.add_error(f"Failed to download GFPGAN model '{model_name}'.")
            return False

        gfpgan_model_path = str(self.gfpgan_model_dir / model_name)
        self.message_manager.add_message(f"Loading GFPGAN face enhancer from {gfpgan_model_path}...")
        try:
            # For the pipeline: Clean Face (at original res) -> Upscale with RealESRGAN
            # GFPGANer `upscale` should be 1.
            self.face_enhancer = GFPGANer(
                model_path=gfpgan_model_path,
                upscale=1, # GFPGAN itself won't further upscale if RealESRGAN handles final scaling.
                arch='clean', 
                channel_multiplier=2,
                bg_upsampler=bg_upsampler, 
                device=self.device
            )
            self.message_manager.add_success("GFPGAN face enhancer loaded.")
            return True
        except Exception as e:
            self.message_manager.add_error(f"Failed to load GFPGAN face enhancer: {e}")
            self.message_manager.add_error(traceback.format_exc())
            self.face_enhancer = None
            return False

    def _unload_face_enhancer(self):
        if self.face_enhancer is not None:
            self.message_manager.add_message("Unloading GFPGAN face enhancer...")
            del self.face_enhancer
            self.face_enhancer = None
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            self.message_manager.add_success("GFPGAN face enhancer unloaded.")
        else:
            self.message_manager.add_message("GFPGAN face enhancer not loaded.")


    def unload_model(self, model_key: str):
        if model_key in self.upsamplers and self.upsamplers[model_key].get("upsampler") is not None:
            config = self.upsamplers.pop(model_key)
            upsampler_instance = config["upsampler"]
            tile_s = config.get("tile_size", 0)
            native_scale = config.get("native_scale", "N/A") # Get native_scale for logging
            log_tile_size = str(tile_s) if tile_s > 0 else "Auto"
            self.message_manager.add_message(f"Unloading ESRGAN model '{model_key}' (Scale: {native_scale}x, Tile: {log_tile_size})...")
            
            if self.face_enhancer and hasattr(self.face_enhancer, 'bg_upsampler') and self.face_enhancer.bg_upsampler == upsampler_instance:
                self.message_manager.add_message("Unloading associated GFPGAN as its BG upsampler is being removed.")
                self._unload_face_enhancer()

            del upsampler_instance 
            devicetorch.empty_cache(torch) 
            gc.collect()
            self.message_manager.add_success(f"ESRGAN model '{model_key}' unloaded and memory cleared.")
        else:
            self.message_manager.add_message(f"ESRGAN model '{model_key}' not loaded, no need to unload.")

    def unload_all_models(self): 
        if not self.upsamplers and not self.face_enhancer:
            self.message_manager.add_message("No ESRGAN or GFPGAN models currently loaded.")
            return
            
        self.message_manager.add_message("Unloading all ESRGAN models...")
        model_keys_to_unload = list(self.upsamplers.keys()) 
        for key in model_keys_to_unload: 
            if key in self.upsamplers: 
                config = self.upsamplers.pop(key)
                upsampler_instance = config["upsampler"]
                del upsampler_instance # type: ignore
        
        self._unload_face_enhancer() 

        devicetorch.empty_cache(torch) 
        gc.collect()
        self.message_manager.add_success("All ESRGAN and GFPGAN models unloaded and memory cleared.")

    def upscale_frame(self, frame_np_array, model_key: str, enhance_face: bool = False):
        """
        Upscales a single frame using the specified model and its native scale.
        The `output_scale_factor` parameter is removed; scale is determined by the model.
        """
        config = self.upsamplers.get(model_key)
        upsampler: RealESRGANer | None = None
        current_tile_size = 0
        model_native_scale = 0 # Will be fetched from config

        if config and config.get("upsampler"):
            upsampler = config["upsampler"] # type: ignore
            current_tile_size = config.get("tile_size", 0) # type: ignore
            model_native_scale = config.get("native_scale", 0) # type: ignore
            if model_native_scale == 0: # Should not happen if model loaded correctly
                self.message_manager.add_error(f"Error: Native scale for model '{model_key}' is 0 or not found in config.")
                return None
        
        if upsampler is None:
            self.message_manager.add_warning(
                f"ESRGAN model '{model_key}' not pre-loaded. Attempting to load now (with default Tile: Auto)..."
            )
            tile_to_load_with = config.get("tile_size", 0) if config else 0
            upsampler = self.load_model(model_key, tile_size=tile_to_load_with)
            if upsampler is None:
                self.message_manager.add_error(f"Failed to auto-load ESRGAN model '{model_key}'. Cannot upscale.")
                return None
            
            # Re-fetch config after load_model to get native_scale and tile_size
            loaded_config = self.upsamplers.get(model_key)
            if loaded_config:
                current_tile_size = loaded_config.get("tile_size", 0) # type: ignore
                model_native_scale = loaded_config.get("native_scale", 0) # type: ignore
                if model_native_scale == 0:
                    self.message_manager.add_error(f"Error: Native scale for auto-loaded model '{model_key}' is 0.")
                    return None
            else: # Should not happen if load_model succeeded
                self.message_manager.add_error(f"Error: Config for auto-loaded model '{model_key}' not found.")
                return None


        if enhance_face:
            if not self.face_enhancer or (hasattr(self.face_enhancer, 'bg_upsampler') and self.face_enhancer.bg_upsampler != upsampler):
                self.message_manager.add_message("Face enhancement requested, loading/re-configuring GFPGAN...")
                self._load_face_enhancer(bg_upsampler=upsampler) 
            
            if not self.face_enhancer: 
                self.message_manager.add_warning("GFPGAN could not be loaded. Proceeding without face enhancement.")
                enhance_face = False 

        try:
            img_bgr = frame_np_array[:, :, ::-1] 
            
            # The `outscale` parameter for `upsampler.enhance` will be the model's native scale.
            # This ensures RealESRGANer does not perform an internal cv2.resize if the
            # requested outscale differs from its own `self.scale`.
            outscale_for_enhance = float(model_native_scale)

            if enhance_face and self.face_enhancer:
                if self.face_enhancer.upscale != 1:
                     self.message_manager.add_warning(
                         f"GFPGANer's internal upscale is {self.face_enhancer.upscale}, but for pipeline 'Clean->Upscale', it should be 1. "
                         "Results might be unexpected."
                     ) # This warning remains valid as a sanity check.
                
                _, _, cleaned_img_bgr = self.face_enhancer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
                # Upscale the cleaned image to the model's native scale
                output_bgr, _ = upsampler.enhance(cleaned_img_bgr, outscale=outscale_for_enhance)
            else:
                # Standard RealESRGAN upscaling to the model's native scale
                output_bgr, _ = upsampler.enhance(img_bgr, outscale=outscale_for_enhance) 
            
            output_rgb = output_bgr[:, :, ::-1] 
            return output_rgb
        except Exception as e:
            tile_size_msg_part = str(current_tile_size) if current_tile_size > 0 else 'Auto'
            face_msg_part = " + Face Enhance" if enhance_face else ""
            # Log the model's native scale as the target scale
            self.message_manager.add_error(f"Error during ESRGAN frame upscaling (Model: {model_key}{face_msg_part}, Target Scale: {model_native_scale}x, Tile: {tile_size_msg_part}): {e}")
            self.message_manager.add_error(traceback.format_exc()) 
            if "out of memory" in str(e).lower() and self.device.type == 'cuda':
                self.message_manager.add_warning(
                    "CUDA OOM during upscaling. Emptying cache. "
                    f"Current model (Model: {model_key}, Tile: {tile_size_msg_part}) may need reloading. "
                    "Consider using a smaller tile size or a smaller input video if issues persist."
                )
                devicetorch.empty_cache(torch)
            return None