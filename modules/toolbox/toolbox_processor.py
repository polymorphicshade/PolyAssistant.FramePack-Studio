import os
import gc
import sys
import re
import numpy as np
import torch
import imageio
import gradio as gr
import subprocess
import devicetorch
import json
import math
import shutil # For moving files

from datetime import datetime
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from torchvision.transforms.functional import to_tensor, to_pil_image

from modules.toolbox.rife_core import RIFEHandler
from modules.toolbox.esrgan_core import ESRGANUpscaler
from modules.toolbox.message_manager import MessageManager

device_name_str = devicetorch.get(torch)

VIDEO_QUALITY = 8 # Used by imageio.mimwrite quality/quantizer

class VideoProcessor:
    def __init__(self, message_manager: MessageManager, settings):
        self.message_manager = message_manager
        self.rife_handler = RIFEHandler(message_manager)
        self.device_obj = torch.device(device_name_str) # Store device_obj
        self.esrgan_upscaler = ESRGANUpscaler(message_manager, self.device_obj)
        self.ffmpeg_exe, self.ffprobe_exe = self._tb_find_ffmpeg_executables()
        self.settings = settings
        
        studio_output_dir = Path(self.settings.get("output_dir"))

        # Define the underlying fixed paths
        self._base_temp_output_dir = studio_output_dir / "postprocessed_output" / "temp_processing"
        self._base_permanent_save_dir = studio_output_dir / "postprocessed_output" / "saved_videos"
        
        # self.toolbox_video_output_dir is the DYNAMIC "switchable" path.
        # It defaults to the temporary location.
        self.toolbox_video_output_dir = self._base_temp_output_dir
        
        # This is always where manual "copy to permanent" goes.
        self.toolbox_permanent_save_dir = self._base_permanent_save_dir 
        
        os.makedirs(self._base_temp_output_dir, exist_ok=True)
        os.makedirs(self._base_permanent_save_dir, exist_ok=True)

        # These remain fixed as they are not affected by autosave for general processing
        self.extracted_frames_target_path = studio_output_dir / "postprocessed_output" / "toolbox_frames" / "extracted_frames"
        os.makedirs(self.extracted_frames_target_path, exist_ok=True)
        self.reassembled_video_target_path = studio_output_dir / "postprocessed_output" / "toolbox_frames" / "reassembled_videos"
        os.makedirs(self.reassembled_video_target_path, exist_ok=True)

    # NEW METHOD to flip the "railroad switch"
    def set_autosave_mode(self, autosave_enabled: bool):
        if autosave_enabled:
            self.toolbox_video_output_dir = self._base_permanent_save_dir
            self.message_manager.add_message("Autosave ENABLED: Processed videos will be saved to the permanent folder.", "SUCCESS")
        else:
            self.toolbox_video_output_dir = self._base_temp_output_dir
            self.message_manager.add_message("Autosave DISABLED: Processed videos will be saved to the temporary folder.", "INFO")

    def _tb_find_ffmpeg_executables(self):
        ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        ffprobe_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
        
        ffmpeg_path, ffprobe_path = None, None

        try:
            imageio_ffmpeg_exe = imageio.plugins.ffmpeg.get_exe()
            if os.path.isfile(imageio_ffmpeg_exe):
                result = subprocess.run([imageio_ffmpeg_exe, "-version"], capture_output=True, text=True, check=False, errors='ignore')
                if result.returncode == 0 and "ffmpeg version" in result.stdout.lower():
                    ffmpeg_path = imageio_ffmpeg_exe
                    self.message_manager.add_message(f"Located FFmpeg via imageio: {ffmpeg_path}", "SUCCESS")
                    potential_ffprobe = os.path.join(os.path.dirname(ffmpeg_path), ffprobe_name)
                    if os.path.isfile(potential_ffprobe):
                        result_probe = subprocess.run([potential_ffprobe, "-version"], capture_output=True, text=True, check=False, errors='ignore')
                        if result_probe.returncode == 0 and "ffprobe version" in result_probe.stdout.lower():
                            ffprobe_path = potential_ffprobe
                            self.message_manager.add_message(f"Located FFprobe via imageio dir: {ffprobe_path}", "SUCCESS")
        except Exception as e:
            self.message_manager.add_message(f"Imageio FFmpeg check failed: {e}. Trying system PATH.", "WARNING")

        if not ffmpeg_path:
            try:
                result = subprocess.run([ffmpeg_name, "-version"], capture_output=True, text=True, check=True, errors='ignore')
                if "ffmpeg version" in result.stdout.lower():
                    ffmpeg_path = ffmpeg_name
                    self.message_manager.add_message(f"Located FFmpeg in system PATH.", "SUCCESS")
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                self.message_manager.add_message(f"FFmpeg not found in system PATH: {e}", "ERROR")
        
        if not ffprobe_path:
            try:
                result = subprocess.run([ffprobe_name, "-version"], capture_output=True, text=True, check=True, errors='ignore')
                if "ffprobe version" in result.stdout.lower():
                    ffprobe_path = ffprobe_name
                    self.message_manager.add_message(f"Located FFprobe in system PATH.", "SUCCESS")
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                 if ffmpeg_path : 
                    self.message_manager.add_message(f"FFprobe not found in system PATH: {e}. Audio features might be limited.", "WARNING")

        if not ffmpeg_path:
             self.message_manager.add_message(
                "Critical: FFmpeg executable could not be found. Video processing will likely fail. "
                "Please install FFmpeg and ensure it's in your system PATH, "
                "or install imageio-ffmpeg: pip install imageio-ffmpeg --upgrade", "ERROR"
            )
        return ffmpeg_path, ffprobe_path
        
    # def _tb_find_ffmpeg_executables(self):
        # self.message_manager.add_message(
            # "DEBUG: _tb_find_ffmpeg_executables INTENTIONALLY RETURNING None, None. "
            # "VideoProcessor will not have ffmpeg/ffprobe.", "DEBUG"
        # )
        # Intentionally do not search for or assign ffmpeg_path or ffprobe_path
        # This means self.ffmpeg_exe and self.ffprobe_exe in VideoProcessor will be None
        # return None, None
    
    def _tb_log_ffmpeg_error(self, e_ffmpeg: subprocess.CalledProcessError, operation_description: str):
        """Helper to log FFmpeg errors consistently."""
        self.message_manager.add_error(f"FFmpeg failed during {operation_description}.")
        ffmpeg_stderr_str = e_ffmpeg.stderr.strip() if e_ffmpeg.stderr else ""
        ffmpeg_stdout_str = e_ffmpeg.stdout.strip() if e_ffmpeg.stdout else ""

        details_log = []
        if ffmpeg_stderr_str: details_log.append(f"FFmpeg Stderr: {ffmpeg_stderr_str}")
        if ffmpeg_stdout_str: details_log.append(f"FFmpeg Stdout: {ffmpeg_stdout_str}")
        
        if details_log:
            self.message_manager.add_message("FFmpeg Output:\n" + "\n".join(details_log), "INFO")
        else:
            self.message_manager.add_message(f"No specific output from FFmpeg. (Return code: {e_ffmpeg.returncode}, Command: '{e_ffmpeg.cmd}')", "INFO")

    def tb_extract_frames(self, video_path, extraction_rate, progress=gr.Progress()):
        if video_path is None:
            self.message_manager.add_warning("No input video for frame extraction.")
            return None
        if not isinstance(extraction_rate, int) or extraction_rate < 1:
            self.message_manager.add_error("Extraction rate must be a positive integer (1 for all frames, N for every Nth frame).")
            return None

        resolved_video_path = str(Path(video_path).resolve())
        output_folder_name = self._tb_generate_output_folder_path(
            resolved_video_path, 
            suffix=f"extracted_every_{extraction_rate}")
        os.makedirs(output_folder_name, exist_ok=True)

        self.message_manager.add_message(
            f"Starting frame extraction for {os.path.basename(resolved_video_path)} (every {extraction_rate} frame(s))."
        )
        self.message_manager.add_message(f"Outputting to: {output_folder_name}")
        progress(0, desc="Initializing frame extraction...")
        
        reader = None 
        try:
            reader = imageio.get_reader(resolved_video_path, 'ffmpeg')
            total_frames = None
            try:
                total_frames = reader.count_frames()
                if total_frames == float('inf'): 
                    total_frames = None 
            except Exception: 
                 meta_nframes = reader.get_meta_data().get('nframes')
                 if meta_nframes and meta_nframes != float('inf'):
                     total_frames = meta_nframes
                 else: 
                    self.message_manager.add_warning("Could not determine total frames for precise progress. Will process until end.")
                    total_frames = None

            extracted_count = 0
            frame_iterable = reader
            if total_frames:
                frame_iterable = progress.tqdm(reader, total=total_frames, desc="Extracting frames")
            else: 
                self.message_manager.add_message("Processing frames (total unknown)...")


            for i, frame in enumerate(frame_iterable):
                if not total_frames and i % 100 == 0: 
                    progress(i / (i + 1000.0), desc=f"Extracting frame {i+1}...") 
                
                if i % extraction_rate == 0:
                    frame_filename = f"frame_{extracted_count:06d}.png"
                    output_frame_path = os.path.join(output_folder_name, frame_filename)
                    imageio.imwrite(output_frame_path, frame, format='PNG')
                    extracted_count += 1
            
            progress(1.0, desc="Extraction complete.")
            self.message_manager.add_success(f"Successfully extracted {extracted_count} frames to: {output_folder_name}")
            return output_folder_name

        except Exception as e:
            self.message_manager.add_error(f"Error during frame extraction: {e}")
            import traceback
            self.message_manager.add_error(traceback.format_exc())
            progress(1.0, desc="Error during extraction.")
            return None
        finally:
            if reader: 
                reader.close()
            gc.collect()
            
    def tb_get_extracted_frame_folders(self) -> list:
        """Scans the extracted_frames_target_path for subdirectories."""
        if not os.path.exists(self.extracted_frames_target_path):
            self.message_manager.add_warning(f"Extracted frames directory not found: {self.extracted_frames_target_path}")
            return []
        try:
            folders = [
                d for d in os.listdir(self.extracted_frames_target_path)
                if os.path.isdir(os.path.join(self.extracted_frames_target_path, d))
            ]
            # Optionally, sort them (e.g., alphabetically or by creation time if desired)
            folders.sort() 
            self.message_manager.add_message(f"Found {len(folders)} extracted frame folders.")
            return folders
        except Exception as e:
            self.message_manager.add_error(f"Error scanning for extracted frame folders: {e}")
            return []

    def tb_delete_extracted_frames_folder(self, folder_name_to_delete: str) -> bool:
        """Deletes a specified subfolder from the extracted_frames_target_path."""
        if not folder_name_to_delete:
            self.message_manager.add_warning("No folder selected for deletion.")
            return False
        
        folder_path_to_delete = os.path.join(self.extracted_frames_target_path, folder_name_to_delete)

        if not os.path.exists(folder_path_to_delete) or not os.path.isdir(folder_path_to_delete):
            self.message_manager.add_error(f"Folder not found or is not a directory: {folder_path_to_delete}")
            return False
        
        try:
            shutil.rmtree(folder_path_to_delete)
            self.message_manager.add_success(f"Successfully deleted folder: {folder_name_to_delete}")
            return True
        except Exception as e:
            self.message_manager.add_error(f"Error deleting folder '{folder_name_to_delete}': {e}")
            self.message_manager.add_error(traceback.format_exc() if 'traceback' in sys.modules else str(e)) # Add traceback if available
            return False
            
    # MODIFIED tb_reassemble_frames_to_video
    def tb_reassemble_frames_to_video(self, frames_source, output_fps, output_base_name_override=None, progress=gr.Progress()):
        if not frames_source:
            self.message_manager.add_warning("No frames source (folder or files) provided for reassembly.")
            return None
        
        try:
            output_fps = int(output_fps)
            if output_fps <= 0:
                self.message_manager.add_error("Output FPS must be a positive number.")
                return None
        except ValueError:
            self.message_manager.add_error("Invalid FPS value for reassembly.")
            return None

        self.message_manager.add_message(f"Starting frame reassembly to video at {output_fps} FPS.")
        
        frame_info_list = []
        frames_data_prepared = False

        try:
            if isinstance(frames_source, str) and os.path.isdir(frames_source):
                self.message_manager.add_message(f"Processing frames from directory: {frames_source}")
                for filename in os.listdir(frames_source):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        full_path = os.path.join(frames_source, filename)
                        frame_info_list.append({
                            'original_like_filename': filename,
                            'temp_path': full_path
                        })
            elif isinstance(frames_source, list):
                self.message_manager.add_message(f"Processing {len(frames_source)} uploaded files for reassembly.")
                for temp_file_wrapper in frames_source:
                    temp_path = temp_file_wrapper.name
                    original_like_filename = os.path.basename(temp_path)
                    frame_info_list.append({
                        'original_like_filename': original_like_filename,
                        'temp_path': temp_path
                    })
            else:
                self.message_manager.add_error("Invalid frames_source type for reassembly.")
                return None
            
            if not frame_info_list:
                self.message_manager.add_warning("No valid image files found in the provided source to reassemble.")
                return None

            def natural_sort_key_for_dict(item):
                filename = item['original_like_filename']
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', filename)]

            frame_info_list.sort(key=natural_sort_key_for_dict)
            self.message_manager.add_message(f"Sorted {len(frame_info_list)} frames based on their filenames.")
            
            if frame_info_list:
                debug_sorted_names = [info['original_like_filename'] for info in frame_info_list[:min(5, len(frame_info_list))]]
                self.message_manager.add_message(f"DEBUG: First {len(debug_sorted_names)} sorted filenames: {debug_sorted_names}")

            output_file_basename = "reassembled_video"
            if output_base_name_override and isinstance(output_base_name_override, str) and output_base_name_override.strip():
                sanitized_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in output_base_name_override.strip())
                output_file_basename = Path(sanitized_name).stem
                if not output_file_basename:
                    output_file_basename = "reassembled_video" 
                self.message_manager.add_message(f"Using custom output video base name: {output_file_basename}")

            output_video_path = self._tb_generate_output_path(
                input_material_name=output_file_basename,
                suffix=f"{output_fps}fps_reassembled", 
                target_dir=self.reassembled_video_target_path,
                ext=".mp4"
            )

            frames_data = []
            frames_data_prepared = True

            self.message_manager.add_message("Reading frame images (in sorted order)...")
            
            frame_iterator = frame_info_list
            # MODIFIED CONDITION: Check frame_info_list first, then progress and hasattr
            if frame_info_list and progress is not None and hasattr(progress, 'tqdm'):
                 frame_iterator = progress.tqdm(frame_info_list, desc="Reading frames")

            for frame_info in frame_iterator:
                frame_actual_path = frame_info['temp_path']
                filename_for_log = frame_info['original_like_filename']
                try:
                    if not filename_for_log.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        self.message_manager.add_warning(f"Skipping non-standard image file: {filename_for_log}.")
                        continue
                    frames_data.append(imageio.imread(frame_actual_path))
                except Exception as e_read_frame: 
                    self.message_manager.add_warning(f"Could not read frame ({filename_for_log}): {e_read_frame}. Skipping.")
            
            if not frames_data:
                self.message_manager.add_error("No valid frames could be successfully read for reassembly.")
                return None

            self.message_manager.add_message(f"Writing {len(frames_data)} frames to video: {output_video_path}")
            
            imageio.mimwrite(output_video_path, frames_data, fps=output_fps, quality=VIDEO_QUALITY, macro_block_size=16)

            self.message_manager.add_success(f"Successfully reassembled {len(frames_data)} frames into: {output_video_path}")
            return output_video_path

        except Exception as e:
            self.message_manager.add_error(f"Error during frame reassembly: {e}")
            import traceback
            self.message_manager.add_error(traceback.format_exc())
            return None
        finally:
            if frames_data_prepared and 'frames_data' in locals(): 
                del frames_data 
            gc.collect()

    def _tb_clean_filename(self, filename):
        filename = re.sub(r'_\d{6}_\d{6}', '', filename)
        filename = re.sub(r'_\d{6}_\d{4}', '', filename) 
        return filename.strip('_')

    def _tb_generate_output_path(self, input_material_name, suffix, target_dir, ext=".mp4"):
        """
        Generates a unique output path for a file within a specified target directory.
        All processed single video files will use self.toolbox_video_output_dir as target_dir.
        Reassembled videos will use self.reassembled_video_target_path.
        """
        base_name = Path(input_material_name).stem 
        if not base_name: 
            base_name = "untitled_video" 
        cleaned_name = self._tb_clean_filename(base_name)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = f"{cleaned_name}_{suffix}_{timestamp}{ext}"
        return os.path.join(target_dir, filename)
    
    def _tb_generate_output_folder_path(self, input_video_path, suffix):
        """
        Generates a unique output path for a *folder* (e.g., for extracted frames).
        This will always target self.extracted_frames_target_path.
        """
        base_name = Path(input_video_path).stem
        if not base_name:
            base_name = "untitled_video_frames"
        cleaned_name = self._tb_clean_filename(base_name)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        folder_name = f"{cleaned_name}_{suffix}_{timestamp}"
        return os.path.join(self.extracted_frames_target_path, folder_name)

    def tb_copy_video_to_permanent_storage(self, temp_video_path):
        if not temp_video_path or not os.path.exists(temp_video_path):
            self.message_manager.add_error("No video file provided or file does not exist to save.")
            return temp_video_path 

        try:
            video_filename = Path(temp_video_path).name
            permanent_video_path = os.path.join(self.toolbox_permanent_save_dir, video_filename)

            # Ensure the permanent save directory exists (should be by __init__, but good practice)
            os.makedirs(self.toolbox_permanent_save_dir, exist_ok=True)

            self.message_manager.add_message(f"Copying '{video_filename}' to permanent storage: '{permanent_video_path}'")
            shutil.copy2(temp_video_path, permanent_video_path) # copy2 preserves metadata
            self.message_manager.add_success(f"Video saved to: {permanent_video_path}")
            return permanent_video_path # Return the new path in the permanent store
        except Exception as e:
            self.message_manager.add_error(f"Error saving video to permanent storage: {e}")
            self.message_manager.add_error(traceback.format_exc())
            return temp_video_path # Return original temp path if copy failed
            
    def tb_analyze_video_input(self, video_path):
        if video_path is None:
            self.message_manager.add_warning("No video provided for analysis.")
            return "Please upload a video."
        try:
            self.message_manager.add_message(f"Analyzing video: {os.path.basename(video_path)}")
            resolved_video_path = str(Path(video_path).resolve())

            analysis_report = "Could not analyze video fully." 
            
            if self.ffprobe_exe:
                try:
                    probe_cmd = [
                        self.ffprobe_exe, "-v", "error", "-show_format", "-show_streams",
                        "-of", "json", resolved_video_path
                    ]
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, errors='ignore')
                    probe_data = json.loads(result.stdout)
                    
                    video_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"), None)
                    audio_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"), None)

                    if not video_stream:
                        self.message_manager.add_error("No video stream found in the file.")
                        return "Error: No video stream found."

                    duration_str = probe_data.get("format", {}).get("duration", "0") 
                    try:
                        duration = float(duration_str)
                    except ValueError:
                        duration = 0.0
                        self.message_manager.add_warning(f"Could not parse duration '{duration_str}' as float, using 0.0.")
                    duration_display = f"{duration:.2f} seconds"

                    r_frame_rate_str = video_stream.get("r_frame_rate", "0/0")
                    avg_frame_rate_str = video_stream.get("avg_frame_rate", "0/0")
                    fps_display = "N/A"
                    calculated_fps = 0.0

                    def parse_fps(fps_s):
                        if isinstance(fps_s, (int, float)): return float(fps_s)
                        if isinstance(fps_s, str) and "/" in fps_s:
                            try:
                                num, den = map(float, fps_s.split('/'))
                                return num / den if den != 0 else 0.0
                            except ValueError: return 0.0
                        try: return float(fps_s) 
                        except ValueError: return 0.0

                    r_fps_val = parse_fps(r_frame_rate_str)
                    avg_fps_val = parse_fps(avg_frame_rate_str)

                    if r_fps_val > 0:
                        calculated_fps = r_fps_val
                        fps_display = f"{r_fps_val:.2f} FPS"
                        if r_frame_rate_str != avg_frame_rate_str and avg_fps_val > 0 and abs(r_fps_val - avg_fps_val) > 0.01:
                            calculated_fps = avg_fps_val 
                            fps_display = f"{avg_fps_val:.2f} FPS (Average)" 
                    elif avg_fps_val > 0: 
                        calculated_fps = avg_fps_val
                        fps_display = f"{avg_fps_val:.2f} FPS (Average)"
                    
                    width = video_stream.get("width", "N/A")
                    height = video_stream.get("height", "N/A")
                    resolution_display = f"{width}x{height}"

                    nframes_str = video_stream.get("nb_frames")
                    nframes_display = "N/A"
                    num_frames_value = None

                    if nframes_str and nframes_str.isdigit():
                        num_frames_value = int(nframes_str)
                        nframes_display = str(num_frames_value)
                    elif duration > 0 and calculated_fps > 0:
                        num_frames_value = int(duration * calculated_fps)
                        nframes_display = f"{num_frames_value} (Calculated)"
                    
                    has_audio_str = "No"
                    if audio_stream:
                        audio_codec = audio_stream.get("codec_name", "N/A")
                        audio_channels = audio_stream.get("channels", "N/A")
                        audio_sample_rate = audio_stream.get("sample_rate", "N/A")
                        has_audio_str = f"Yes (Codec: {audio_codec}, Channels: {audio_channels}, Rate: {audio_sample_rate} Hz)"

                    analysis_report = (
                        f"Video Analysis for: {os.path.basename(video_path)}\n"
                        f"------------------------------------\n"
                        f"Duration: {duration_display}\n"
                        f"Frame Rate: {fps_display}\n"
                        f"Resolution: {resolution_display}\n"
                        f"Frames: {nframes_display}\n"
                        f"Audio: {has_audio_str}\n"
                        f"Source: {video_path}"
                    )
                    self.message_manager.add_success("Video analysis complete (using ffprobe).")
                    return analysis_report

                except subprocess.CalledProcessError as e_ffprobe_call:
                    self.message_manager.add_warning(f"ffprobe analysis failed, falling back to imageio.")
                    self._tb_log_ffmpeg_error(e_ffprobe_call, "video analysis with ffprobe")
                except json.JSONDecodeError as e_json:
                    self.message_manager.add_warning(f"ffprobe output was not valid JSON ({e_json}), falling back to imageio.")
                except Exception as e_ffprobe:
                    self.message_manager.add_warning(f"ffprobe analysis failed ({e_ffprobe}), falling back to imageio.")
            
            self.message_manager.add_message("Attempting analysis with imageio...")
            reader = imageio.get_reader(resolved_video_path, 'ffmpeg')
            meta = reader.get_meta_data()
            duration_imgio = meta.get('duration', 'N/A')
            fps_val_imgio = meta.get('fps', 'N/A') 
            size_imgio = meta.get('size', ('N/A', 'N/A'))
            nframes_val_imgio_meta = meta.get('nframes', "N/A") 
            nframes_val_imgio_display = "N/A"

            if nframes_val_imgio_meta not in [float('inf'), "N/A", None] and isinstance(nframes_val_imgio_meta, (int,float)):
                nframes_val_imgio_display = str(int(nframes_val_imgio_meta))
            else:
                try: 
                    nframes_val_imgio_count = reader.count_frames()
                    if nframes_val_imgio_count != float('inf'):
                         nframes_val_imgio_display = f"{int(nframes_val_imgio_count)} (Counted)"
                    else:
                         nframes_val_imgio_display = "Unknown (Stream or very long)"
                except Exception as e_count: 
                    self.message_manager.add_warning(f"imageio reader.count_frames() failed: {e_count}")
                    nframes_val_imgio_display = "Unknown (Frame count failed)"
            reader.close()

            resolution_display_imgio = "N/A"
            if isinstance(size_imgio, tuple) and len(size_imgio) == 2:
                resolution_display_imgio = f"{size_imgio[0]}x{size_imgio[1]}"
            
            fps_display_imgio = f"{fps_val_imgio} FPS" if fps_val_imgio != 'N/A' else "N/A"


            analysis_report = (
                f"Video Analysis for: {os.path.basename(video_path)} (imageio)\n"
                f"------------------------------------\n"
                f"Duration: {duration_imgio} seconds\n"
                f"Frame Rate: {fps_display_imgio}\n"
                f"Resolution: {resolution_display_imgio}\n"
                f"Frames: {nframes_val_imgio_display}\n"
                f"Source: {video_path}"
            )
            self.message_manager.add_success("Video analysis complete (using imageio).")
            return analysis_report
        except Exception as e:
            self.message_manager.add_error(f"Error analyzing video: {e}")
            import traceback
            self.message_manager.add_error(traceback.format_exc())
            return f"Error analyzing video: {e}"

    def _tb_has_audio_stream(self, video_path_to_check):
        if not self.ffprobe_exe:
            self.message_manager.add_warning("FFprobe not found, cannot determine if video has audio. Assuming no audio for ping-pong audio handling.")
            return False
        try:
            resolved_path = str(Path(video_path_to_check).resolve())
            ffprobe_cmd = [
                self.ffprobe_exe, "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0", resolved_path
            ]
            audio_check_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=False, errors='ignore') 

            if audio_check_result.returncode == 0 and "audio" in audio_check_result.stdout.strip().lower():
                return True
            else:
                return False
        except FileNotFoundError: 
            self.message_manager.add_warning("FFprobe executable not found during audio stream check. Assuming no audio.")
            return False
        except Exception as e:
            self.message_manager.add_warning(f"Error checking for audio stream in {os.path.basename(video_path_to_check)}: {e}. Assuming no audio.")
            return False
            
    def tb_process_frames(self, video_path, target_fps_mode, speed_factor, progress=gr.Progress()):
        if video_path is None:
            self.message_manager.add_warning("No input video for frame processing.")
            return None
        if not self.ffmpeg_exe :
            self.message_manager.add_error("FFmpeg executable not found. Cannot process video.")
            return None
        
        final_output_path = None 

        try:
            self.message_manager.add_message(
                f"Starting frame processing for {os.path.basename(video_path)}: "
                f"FPS Mode: {target_fps_mode}, Speed: {speed_factor}x"
            )
            progress(0, desc="Initializing...")
            resolved_video_path = str(Path(video_path).resolve())

            self.message_manager.add_message("Reading video frames...")
            progress(0.05, desc="Reading video...")
            reader = imageio.get_reader(resolved_video_path, 'ffmpeg')
            original_fps = reader.get_meta_data()['fps']
            video_frames = [frame for frame in reader]
            reader.close()
            self.message_manager.add_message(f"Read {len(video_frames)} frames at {original_fps} FPS.")

            processed_frames = video_frames
            current_fps = original_fps

            if speed_factor != 1.0:
                self.message_manager.add_message(f"Adjusting speed by {speed_factor}x (frame sampling/duplication)...")
                progress(0.2, desc="Adjusting speed...")
                if speed_factor > 1.0: 
                    indices = np.arange(0, len(video_frames), speed_factor).astype(int)
                    processed_frames = [video_frames[i] for i in indices if i < len(video_frames)]
                else: 
                    new_len = int(len(video_frames) / speed_factor)
                    indices = np.linspace(0, len(video_frames) - 1, new_len).astype(int)
                    processed_frames = [video_frames[i] for i in indices]
                self.message_manager.add_message(f"Speed adjustment (sampling) resulted in {len(processed_frames)} frames.")
            
            should_interpolate = (target_fps_mode == "2x RIFE Interpolation")
            
            if should_interpolate and len(processed_frames) > 1:
                self.message_manager.add_message("Attempting to load RIFE model for 2x interpolation...")
                if not self.rife_handler._ensure_model_downloaded_and_loaded(): # RIFEHandler internal method
                    self.message_manager.add_error("RIFE model could not be loaded. Skipping interpolation.")
                else:
                    self.message_manager.add_message("RIFE model loaded. Starting RIFE 2x interpolation...")
                    interpolated_video_frames = []
                    num_pairs = len(processed_frames) - 1
                    for i in progress.tqdm(range(num_pairs), desc="RIFE Interpolating (2x)"):
                        frame1_np = processed_frames[i]
                        frame2_np = processed_frames[i+1]
                        
                        interpolated_video_frames.append(frame1_np) 
                        middle_frame_np = self.rife_handler.interpolate_between_frames(frame1_np, frame2_np)
                        
                        if middle_frame_np is not None:
                            interpolated_video_frames.append(middle_frame_np)
                        else:
                            self.message_manager.add_warning(f"Interpolation failed for frame pair {i}. Duplicating previous frame.")
                            interpolated_video_frames.append(frame1_np) 
                    
                    interpolated_video_frames.append(processed_frames[-1]) 
                    processed_frames = interpolated_video_frames
                    current_fps = original_fps * 2 
                    self.message_manager.add_message(f"RIFE 2x interpolation resulted in {len(processed_frames)} frames.")
            
            elif should_interpolate and len(processed_frames) <= 1:
                self.message_manager.add_warning("Not enough frames for RIFE interpolation. Skipping.")

            op_suffix_parts = []
            if speed_factor != 1.0: op_suffix_parts.append(f"speed{speed_factor:.2f}x".replace('.',',')) 
            if should_interpolate and self.rife_handler.rife_model is not None: 
                 op_suffix_parts.append("RIFE2x")
            
            if not op_suffix_parts: op_suffix_parts.append("processed") 
            
            op_suffix = "_".join(op_suffix_parts)

            temp_video_suffix = f"{op_suffix}_temp_video"
            video_stream_output_path = self._tb_generate_output_path(
                resolved_video_path, 
                suffix=temp_video_suffix,
                target_dir=self.toolbox_video_output_dir
            )
            
            final_muxed_output_path = video_stream_output_path.replace("_temp_video", "")

            self.message_manager.add_message(f"Saving video stream to {video_stream_output_path} at {current_fps} FPS...")
            progress(0.85, desc="Saving video stream...")
            imageio.mimwrite(video_stream_output_path, processed_frames, fps=current_fps, quality=VIDEO_QUALITY, macro_block_size=16)

            final_output_path = final_muxed_output_path 

            has_audio = False
            if self.ffprobe_exe:
                try:
                    ffprobe_cmd = [
                        self.ffprobe_exe, "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=codec_type", "-of", "csv=p=0", resolved_video_path
                    ]
                    audio_check_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True, errors='ignore')
                    has_audio = "audio" in audio_check_result.stdout.strip().lower()
                except subprocess.CalledProcessError as e_probe:
                    self.message_manager.add_warning(f"FFprobe check for audio failed.")
                    self._tb_log_ffmpeg_error(e_probe, "audio stream check with FFprobe")
                    has_audio = False 
                except FileNotFoundError:
                     self.message_manager.add_warning(f"FFprobe not found during audio check. Assuming no audio.")
                except Exception as e_probe_other:
                    self.message_manager.add_warning(f"FFprobe check for audio failed with other error: {e_probe_other}. Assuming no audio.")


            if has_audio:
                self.message_manager.add_message("Original video has audio. Processing audio with FFmpeg...")
                progress(0.9, desc="Processing audio...")
                ffmpeg_mux_cmd = [self.ffmpeg_exe, "-y", "-loglevel", "error", "-i", video_stream_output_path]
                
                audio_filters = []
                if speed_factor != 1.0:
                    current_atempo_speed = speed_factor 
                    temp_speed = current_atempo_speed
                    while temp_speed > 100.0:
                        audio_filters.append("atempo=100.0")
                        temp_speed /= 100.0
                    while temp_speed < 0.5 and temp_speed > 0: 
                        audio_filters.append("atempo=0.5")
                        temp_speed /= 0.5
                    
                    if temp_speed != 1.0 and (0.5 <= temp_speed <= 100.0):
                        audio_filters.append(f"atempo={temp_speed:.4f}")
                    self.message_manager.add_message(f"Applying audio speed adjustment with atempo: {','.join(audio_filters) if audio_filters else 'None'}")

                ffmpeg_mux_cmd.extend(["-i", resolved_video_path])
                ffmpeg_mux_cmd.extend(["-c:v", "copy"]) 
                
                if audio_filters:
                    ffmpeg_mux_cmd.extend(["-filter:a", ",".join(audio_filters)])
                    ffmpeg_mux_cmd.extend(["-c:a", "aac", "-b:a", "192k"]) 
                else:
                    # Ensure AAC for compatibility even if no speed change
                    self.message_manager.add_message(f"Audio stream found. Re-encoding to AAC for MP4 compatibility (no speed change).", "INFO")
                    ffmpeg_mux_cmd.extend(["-c:a", "aac", "-b:a", "192k"]) 

                ffmpeg_mux_cmd.extend(["-map", "0:v:0", "-map", "1:a:0?", "-shortest", final_muxed_output_path])
                
                try:
                    subprocess.run(ffmpeg_mux_cmd, check=True, capture_output=True, text=True)
                    self.message_manager.add_success(f"Video saved with processed audio: {final_muxed_output_path}")
                    final_output_path = final_muxed_output_path
                except subprocess.CalledProcessError as e_mux:
                    self._tb_log_ffmpeg_error(e_mux, "audio processing/muxing")
                    self.message_manager.add_message("Saving video without audio as fallback.")
                    if os.path.exists(final_muxed_output_path): os.remove(final_muxed_output_path) 
                    os.rename(video_stream_output_path, final_muxed_output_path) 
                    final_output_path = final_muxed_output_path
                except FileNotFoundError: 
                    self.message_manager.add_error(f"FFmpeg not found during muxing. This is unexpected.")
                    if os.path.exists(final_muxed_output_path): os.remove(final_muxed_output_path)
                    os.rename(video_stream_output_path, final_muxed_output_path)
                    final_output_path = final_muxed_output_path
            else: 
                self.message_manager.add_message("No audio in original or audio processing skipped. Saving video-only.")
                if os.path.exists(final_muxed_output_path) and final_muxed_output_path != video_stream_output_path : 
                    os.remove(final_muxed_output_path) 
                os.rename(video_stream_output_path, final_muxed_output_path)
                final_output_path = final_muxed_output_path


            if os.path.exists(video_stream_output_path) and video_stream_output_path != final_muxed_output_path:
                try: 
                    os.remove(video_stream_output_path)
                    self.message_manager.add_message(f"Removed temporary video stream: {video_stream_output_path}")
                except Exception as e_clean: 
                    self.message_manager.add_warning(f"Could not remove temp video file {video_stream_output_path}: {e_clean}")
            
            progress(1.0, desc="Complete.")
            self.message_manager.add_success(f"Frame processing complete: {final_output_path}")
            return final_output_path

        except Exception as e:
            self.message_manager.add_error(f"Error during frame processing: {e}")
            import traceback
            self.message_manager.add_error(traceback.format_exc())
            progress(1.0, desc="Error.")
            return None 
        finally:
            if self.rife_handler: 
                self.rife_handler.unload_model()
            devicetorch.empty_cache(torch) 
            gc.collect()

    def tb_create_loop(self, video_path, loop_type, num_loops, progress=gr.Progress()):
        if video_path is None: self.message_manager.add_warning("No input video for loop creation."); return None
        if not self.ffmpeg_exe: self.message_manager.add_error("FFmpeg not found."); return None
        if loop_type == "none": self.message_manager.add_message("Loop type 'none'. No action."); return video_path

        progress(0, desc="Initializing loop creation...")
        resolved_video_path = str(Path(video_path).resolve())
        output_path = self._tb_generate_output_path(
            resolved_video_path, 
            suffix=f"looped_{loop_type}_{num_loops}x",
            target_dir=self.toolbox_video_output_dir
        )
        
        self.message_manager.add_message(f"Creating {loop_type} ({num_loops}x) for {os.path.basename(resolved_video_path)}...")
        
        ping_pong_unit_path = None 

        try:
            progress(0.2, desc=f"Preparing {loop_type} loop...")
            if loop_type == "ping-pong":
                ping_pong_unit_path = self._tb_generate_output_path(
                    resolved_video_path, 
                    suffix="pingpong_unit_temp", 
                    target_dir=self.toolbox_video_output_dir
                )

                ffmpeg_pp_unit_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-i", resolved_video_path,
                    "-vf", "split[main][tmp];[tmp]reverse[rev];[main][rev]concat=n=2:v=1:a=0",
                    "-an", # Create video-only unit first
                    ping_pong_unit_path
                ]
                subprocess.run(ffmpeg_pp_unit_cmd, check=True, capture_output=True, text=True)
                self.message_manager.add_message(f"Created ping-pong unit (video-only): {ping_pong_unit_path}")

                has_audio_in_original = self._tb_has_audio_stream(resolved_video_path)

                if has_audio_in_original:
                    self.message_manager.add_message("Original video has audio. Will loop audio for ping-pong.")
                    # For ping-pong, the audio duration should match the doubled video unit.
                    # If original video is T seconds, ping-pong unit is 2T seconds.
                    # We need to loop original audio to cover 2T * num_loops.
                    # `aloop=loop={N}` means N additional loops, so N+1 total plays.
                    # The total plays of the original audio needed is (2 * num_loops) - 1 additional loops.
                    audio_loop_count_for_ffmpeg = (num_loops * 2) -1 
                    ffmpeg_cmd = [
                        self.ffmpeg_exe, "-y", "-loglevel", "error",
                        "-stream_loop", str(num_loops - 1), # Loop the video-only ping-pong unit N-1 times (total N plays)
                        "-i", ping_pong_unit_path,
                        "-i", resolved_video_path, # Original video for audio source
                        "-c:v", "copy",
                        "-filter_complex", f"[1:a]aloop=loop={audio_loop_count_for_ffmpeg}:size=2147483647[a_looped]",
                        "-map", "0:v:0",
                        "-map", "[a_looped]",
                        "-c:a", "aac", "-b:a", "192k",
                        "-shortest", # Ensure output duration is based on the shorter of (looped video) and (looped audio)
                        output_path
                    ]
                else:
                    self.message_manager.add_message("Original video has no audio or audio check failed. Creating video-only ping-pong loop.")
                    ffmpeg_cmd = [
                        self.ffmpeg_exe, "-y", "-loglevel", "error",
                        "-stream_loop", str(num_loops - 1), # Loop the video-only unit
                        "-i", ping_pong_unit_path,
                        "-c:v", "copy",
                        "-an", # No audio
                        output_path
                    ]

            else: # Regular 'loop'
                ffmpeg_stream_loop_value = num_loops 
                
                self.message_manager.add_message(f"Regular loop: original video + {ffmpeg_stream_loop_value} additional loop(s).")
                
                ffmpeg_cmd_parts = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-stream_loop", str(ffmpeg_stream_loop_value),
                    "-i", resolved_video_path,
                    "-c:v", "copy" 
                ]

                has_audio_in_original_for_loop = self._tb_has_audio_stream(resolved_video_path)
                if has_audio_in_original_for_loop:
                    self.message_manager.add_message("Original video has audio. Re-encoding to AAC for looped MP4.", "INFO")
                    ffmpeg_cmd_parts.extend([
                        "-c:a", "aac",      
                        "-b:a", "192k",     
                        "-map", "0:v:0",    
                        "-map", "0:a:0?"    
                    ])
                else:
                    self.message_manager.add_message("No audio in original or audio check failed. Looped video will be silent.", "INFO")
                    ffmpeg_cmd_parts.extend([
                        "-an",              
                        "-map", "0:v:0"     
                    ])
                
                ffmpeg_cmd_parts.append(output_path)
                ffmpeg_cmd = ffmpeg_cmd_parts
            
            self.message_manager.add_message(f"Processing video {loop_type} with FFmpeg...")
            progress(0.5, desc=f"Running FFmpeg for {loop_type}...")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, errors='ignore')

            progress(1.0, desc=f"{loop_type.capitalize()} loop created successfully.")
            self.message_manager.add_success(f"Loop creation complete: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e_loop:
            self._tb_log_ffmpeg_error(e_loop, f"{loop_type} creation")
            progress(1.0, desc=f"Error creating {loop_type}.")
            return None
        except Exception as e:
            self.message_manager.add_error(f"Error creating loop: {e}")
            import traceback 
            self.message_manager.add_error(traceback.format_exc())
            progress(1.0, desc="Error creating loop.")
            return None
        finally:
            if ping_pong_unit_path and os.path.exists(ping_pong_unit_path):
                try: 
                    os.remove(ping_pong_unit_path)
                    self.message_manager.add_message(f"Cleaned up temporary ping-pong unit: {ping_pong_unit_path}", "DEBUG")
                except Exception as e_clean_pp: self.message_manager.add_warning(f"Could not remove temp ping-pong unit: {e_clean_pp}")
            gc.collect()

    def _tb_get_video_dimensions(self, video_path):
        video_width = 0
        video_height = 0
        reader = None
        try:
            reader = imageio.get_reader(video_path, 'ffmpeg')
            meta = reader.get_meta_data()
            size_imgio = meta.get('size')  
            if size_imgio and isinstance(size_imgio, tuple) and len(size_imgio) == 2:
                video_width, video_height = int(size_imgio[0]), int(size_imgio[1])
            else:
                self.message_manager.add_warning("Could not get video dimensions for vignette from imageio metadata, defaulting aspect to 1/1.")
        except Exception as e_meta:
            self.message_manager.add_warning(f"Error getting video dimensions for vignette: {e_meta}. Defaulting aspect to 1/1.")
        finally:
            if reader:
                try:
                    reader.close()
                except Exception: 
                    pass
        return video_width, video_height
        
    def _tb_create_vignette_filter(self, strength_percent, width, height):
        min_angle_rad = math.pi / 3.5 
        max_angle_rad = math.pi / 2    
        
        normalized_strength = strength_percent / 100.0 
        angle_rad = min_angle_rad + normalized_strength * (max_angle_rad - min_angle_rad)
        
        vignette_aspect_ratio_val = "1/1" 
        if width > 0 and height > 0:
            vignette_aspect_ratio_val = f"{width/height:.4f}" 

        return f"vignette=angle={angle_rad:.4f}:mode=forward:eval=init:aspect={vignette_aspect_ratio_val}"

    def tb_apply_filters(self, video_path, brightness, contrast, saturation, temperature,
                      sharpen, blur, denoise, vignette, s_curve_contrast, film_grain_strength,
                      progress=gr.Progress()):
        if video_path is None: self.message_manager.add_warning("No input video for filters."); return None
        if not self.ffmpeg_exe: self.message_manager.add_error("FFmpeg not found."); return None

        progress(0, desc="Initializing filter application...")
        resolved_video_path = str(Path(video_path).resolve())
        output_path = self._tb_generate_output_path(
            resolved_video_path, 
            suffix="filtered", 
            target_dir=self.toolbox_video_output_dir)
        self.message_manager.add_message(f"🎨 Applying filters to {os.path.basename(resolved_video_path)}...")

        if vignette > 0:
            video_width, video_height = self._tb_get_video_dimensions(resolved_video_path)
            if video_width > 0 and video_height > 0: 
                 self.message_manager.add_message(f"Video dimensions for vignette: {video_width}x{video_height}", "DEBUG")
            
        filters = []
        applied_filter_descriptions = [] 

        if denoise > 0: 
            ls = denoise * 0.8 
            cs = denoise * 0.6 
            lt = denoise * 0.7  
            ct = denoise * 0.5
            filters.append(f"hqdn3d={ls:.1f}:{cs:.1f}:{lt:.1f}:{ct:.1f}")
            applied_filter_descriptions.append(f"Denoise (hqdn3d {ls:.1f}:{cs:.1f}:{lt:.1f}:{ct:.1f})")
        
        if temperature != 0: 
            mid_shift_val = (temperature / 100.0) * 0.3 
            filters.append(f"colorbalance=rm={mid_shift_val:.2f}:bm={-mid_shift_val:.2f}") 
            applied_filter_descriptions.append(f"Color Temp (Midtones R:{mid_shift_val:+.2f}, B:{-mid_shift_val:+.2f})")

        eq_parts = []
        desc_eq = []
        if brightness != 0: 
            eq_parts.append(f"brightness={brightness/100.0:.2f}")
            desc_eq.append(f"Brightness ({brightness/100.0:+.2f})")
        if contrast != 1: 
            eq_parts.append(f"contrast={contrast:.2f}")
            desc_eq.append(f"Contrast (Linear) ({contrast:.2f}x)")
        if saturation != 1: 
            eq_parts.append(f"saturation={saturation:.2f}")
            desc_eq.append(f"Saturation ({saturation:.2f}x)")
        if eq_parts:
            filters.append(f"eq={':'.join(eq_parts)}")
            applied_filter_descriptions.append(" & ".join(desc_eq))
        
        if s_curve_contrast > 0: 
            s = s_curve_contrast / 100.0 
            y1_mod = 0.25 - s * (0.25 - 0.10) 
            y2_mod = 0.75 + s * (0.90 - 0.75)
            curve_str = f"0/0 0.25/{y1_mod:.2f} 0.75/{y2_mod:.2f} 1/1"
            filters.append(f"curves=all='{curve_str}'")
            applied_filter_descriptions.append(f"S-Curve Contrast (Strength: {s_curve_contrast}%)")

        if blur > 0: 
            blur_sigma = blur * 0.4 
            filters.append(f"gblur=sigma={blur_sigma:.1f}")
            applied_filter_descriptions.append(f"Blur (sigma: {blur_sigma:.1f})")
        
        if sharpen > 0: 
            sharpen_amount = sharpen * 0.3 
            filters.append(f"unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount={sharpen_amount:.2f}")
            applied_filter_descriptions.append(f"Sharpen (amount: {sharpen_amount:.2f})")
        
        if film_grain_strength > 0: 
            actual_noise_strength = film_grain_strength * 0.5 
            filters.append(f"noise=alls={actual_noise_strength:.1f}:allf=t+u") 
            applied_filter_descriptions.append(f"Film Grain (Strength: {actual_noise_strength:.1f})")

        if vignette > 0: 
            filters.append(self._tb_create_vignette_filter(vignette, video_width, video_height))
            applied_filter_descriptions.append(f"Vignette (strength: {vignette}%)")

        if not filters: 
            self.message_manager.add_message("ℹ️ No filters selected. Skipping filter application.")
            progress(1.0, desc="No filters applied.")
            return video_path

        if applied_filter_descriptions:
            self.message_manager.add_message("🔧 Applying FFmpeg filters:")
            for desc_item_filter in applied_filter_descriptions: 
                self.message_manager.add_message(f"  • {desc_item_filter}")
        
        progress(0.2, desc="Preparing filter command...")
        try:
            ffmpeg_cmd_base = [
                self.ffmpeg_exe, "-y", 
                "-loglevel", "error", 
                "-i", resolved_video_path,
                "-vf", ",".join(filters), 
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-map", "0:v:0" # Always map video from input 0
            ]

            audio_handling_opts = []
            if self._tb_has_audio_stream(resolved_video_path):
                self.message_manager.add_message("Original video has audio. Re-encoding to AAC for filtered video.", "INFO")
                audio_handling_opts = [
                    "-c:a", "aac", 
                    "-b:a", "192k",
                    "-map", "0:a:0?" # Map audio from input 0 (optional)
                ]
            else:
                self.message_manager.add_message("No audio in original or audio check failed. Filtered video will be silent.", "INFO")
                audio_handling_opts = ["-an"] # No audio output
            
            ffmpeg_cmd = ffmpeg_cmd_base + audio_handling_opts + [output_path]
            self.message_manager.add_message("🔄 Processing filters with FFmpeg...")
            progress(0.5, desc="Running FFmpeg for filters...")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, errors='ignore') 
            
            progress(1.0, desc="Filters applied successfully.")
            self.message_manager.add_success(f"✅ Filters applied successfully!")
            self.message_manager.add_message(f"💾 Output: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e_filters:
            self._tb_log_ffmpeg_error(e_filters, "filter application")
            progress(1.0, desc="Error applying filters.")
            return None
        except Exception as e:
            self.message_manager.add_error(f"❌ An unexpected error occurred while applying filters: {e}")
            import traceback
            self.message_manager.add_error(traceback.format_exc())
            progress(1.0, desc="Error applying filters.")
            return None
        finally:
            gc.collect()
            

    def tb_upscale_video(self, video_path, upscale_factor_str, progress=gr.Progress()):
        if video_path is None:
            self.message_manager.add_warning("No input video for upscaling.")
            return None
        if not self.ffmpeg_exe:
            self.message_manager.add_error("FFmpeg executable not found. Cannot process audio for upscaled video.")
            return None

        upscale_factor = 0
        final_output_path = None
        reader = None

        try:
            progress(0.0, desc="Parsing upscale factor...")

            try:
                upscale_factor = int(upscale_factor_str.replace('x', ''))
            except ValueError:
                self.message_manager.add_error(f"Invalid upscale factor string: '{upscale_factor_str}'. Must be like '2x' or '4x'.")
                raise 

            if upscale_factor not in [2, 4]:
                self.message_manager.add_error(f"Unsupported upscale factor: {upscale_factor}x. Choose 2x or 4x.")
                return None 

            progress(0.05, desc=f"Loading ESRGAN model ({upscale_factor}x)...")
            self.message_manager.add_message(f"Attempting to load ESRGAN model for {upscale_factor}x upscaling...")
            upsampler_instance = self.esrgan_upscaler.load_model(target_scale=upscale_factor)

            if upsampler_instance is None:
                self.message_manager.add_error(f"Could not load ESRGAN model for {upscale_factor}x. Aborting upscale.")
                return None 

            self.message_manager.add_message(f"ESRGAN model for {upscale_factor}x loaded.")
            progress(0.1, desc="Initializing upscaling process...")
            
            self.message_manager.add_message(
                f"Starting video upscaling for {os.path.basename(video_path)} by {upscale_factor}x."
            )
            resolved_video_path = str(Path(video_path).resolve())
            upscaled_frames = []
            
            progress(0.12, desc="Reading video info...")
            reader = imageio.get_reader(resolved_video_path, 'ffmpeg')
            meta_data = reader.get_meta_data()
            
            original_fps_from_meta = meta_data.get('fps', 30) 
            original_fps = 30.0 
            if isinstance(original_fps_from_meta, (int, float)):
                original_fps = float(original_fps_from_meta)
            elif isinstance(original_fps_from_meta, str):
                try:
                    if "/" in original_fps_from_meta: 
                        num, den = map(float, original_fps_from_meta.split('/'))
                        if den != 0: original_fps = num / den
                    else:
                        original_fps = float(original_fps_from_meta)
                except ValueError:
                    self.message_manager.add_warning(f"Could not parse FPS string '{original_fps_from_meta}', using default 30.0.")
            
            n_frames = None 
            n_frames_from_meta = meta_data.get('nframes')

            if n_frames_from_meta is not None and n_frames_from_meta != float('inf'):
                try:
                    n_frames = int(n_frames_from_meta)
                except (ValueError, TypeError):
                    n_frames = None 

            if n_frames is None: 
                try:
                    counted_frames = reader.count_frames()
                    if counted_frames != float('inf'):
                        n_frames = int(counted_frames)
                except Exception as e_count:
                    self.message_manager.add_warning(f"Could not count frames via imageio: {e_count}. Frame count will be unknown.")
            
            n_frames_display_upscale = str(n_frames) if n_frames is not None else "Unknown"
            self.message_manager.add_message(f"Original FPS: {original_fps:.2f}. Total frames: {n_frames_display_upscale}.")

            progress(0.15, desc="Preparing to upscale frames...")

            processed_frame_count = 0
            base_frame_iterator = enumerate(reader) 
            iterable_for_loop = base_frame_iterator

            if n_frames is not None: 
                iterable_for_loop = progress.tqdm(base_frame_iterator, total=n_frames, desc="Upscaling Frames")
            else:
                self.message_manager.add_message("Total frames unknown, progress bar will update per frame batch.")
                progress(0.15, desc="Upscaling frames (total unknown)...") 

            for i, frame_np in iterable_for_loop:
                processed_frame_count += 1
                if n_frames is None and i % 10 == 0: 
                    loop_progress_span = 0.8 - 0.15 
                    iteration_estimate = (i / (i + 500.0)) 
                    current_overall_progress = 0.15 + (iteration_estimate * loop_progress_span)
                    progress(min(current_overall_progress, 0.79), desc=f"Upscaling frame {processed_frame_count}...")

                upscaled_frame_np = self.esrgan_upscaler.upscale_frame(frame_np, target_scale=upscale_factor)

                if upscaled_frame_np is not None:
                    upscaled_frames.append(upscaled_frame_np)
                else:
                    self.message_manager.add_error(f"Failed to upscale frame {processed_frame_count}. Skipping.")
                    if any("out of memory" in msg_content.lower() for msg_type, msg_content in self.message_manager.messages_history if msg_type == "ERROR"):
                        self.message_manager.add_error("CUDA OOM likely detected. Aborting video upscale.")
                        return None 

                if processed_frame_count > 0 and processed_frame_count % 20 == 0:
                    devicetorch.empty_cache(torch); gc.collect()
            
            if reader:
                try: 
                    reader.close()
                    reader = None 
                except Exception as e_close_early:
                    self.message_manager.add_warning(f"Error closing reader early: {e_close_early}")

            if not upscaled_frames:
                self.message_manager.add_error("No frames were successfully upscaled. Aborting.")
                return None 
            
            self.message_manager.add_message(f"Successfully upscaled {len(upscaled_frames)} frames.")
            progress(0.80, desc="Saving upscaled video stream...")

            temp_video_suffix = f"upscaled_{upscale_factor}x_temp_video"
            video_stream_output_path = self._tb_generate_output_path(
                resolved_video_path,
                suffix=temp_video_suffix,
                target_dir=self.toolbox_video_output_dir
            )
            final_muxed_output_path = video_stream_output_path.replace("_temp_video", "")

            imageio.mimwrite(video_stream_output_path, upscaled_frames, fps=original_fps, quality=VIDEO_QUALITY, macro_block_size=16)
            
            del upscaled_frames; devicetorch.empty_cache(torch); gc.collect()

            has_audio = False
            if self.ffprobe_exe:
                try:
                    ffprobe_cmd = [
                        self.ffprobe_exe, "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=codec_type", "-of", "csv=p=0", resolved_video_path
                    ]
                    audio_check_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True, errors='ignore')
                    has_audio = "audio" in audio_check_result.stdout.strip().lower()
                except subprocess.CalledProcessError as e_probe:
                    self.message_manager.add_warning(f"FFprobe check for audio failed during upscaling.")
                    self._tb_log_ffmpeg_error(e_probe, "audio stream check for upscaling")
                except Exception as e_probe_other:
                    self.message_manager.add_warning(f"FFprobe check for audio failed: {e_probe_other}. Assuming no audio.")

            if has_audio:
                progress(0.90, desc="Muxing audio...")
                self.message_manager.add_message("Original video has audio. Muxing audio...")
                ffmpeg_mux_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-i", video_stream_output_path,
                    "-i", resolved_video_path,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v:0", "-map", "1:a:0?", "-shortest", final_muxed_output_path
                ]
                try:
                    subprocess.run(ffmpeg_mux_cmd, check=True, capture_output=True, text=True)
                    self.message_manager.add_success(f"Upscaled video saved with audio: {final_muxed_output_path}")
                    final_output_path = final_muxed_output_path
                except subprocess.CalledProcessError as e_mux:
                    self._tb_log_ffmpeg_error(e_mux, "audio muxing for upscaled video")
                    self.message_manager.add_message("Saving video without audio as fallback.")
                    if os.path.exists(final_muxed_output_path): os.remove(final_muxed_output_path)
                    os.rename(video_stream_output_path, final_muxed_output_path)
                    final_output_path = final_muxed_output_path
                except FileNotFoundError:
                    self.message_manager.add_error(f"FFmpeg not found during muxing.")
                    if os.path.exists(final_muxed_output_path): os.remove(final_muxed_output_path)
                    os.rename(video_stream_output_path, final_muxed_output_path)
                    final_output_path = final_muxed_output_path
            else: 
                self.message_manager.add_message("No audio in original or audio processing skipped. Saving video-only.")
                if os.path.exists(final_muxed_output_path) and final_muxed_output_path != video_stream_output_path:
                    os.remove(final_muxed_output_path)
                os.rename(video_stream_output_path, final_muxed_output_path)
                final_output_path = final_muxed_output_path

            if os.path.exists(video_stream_output_path) and video_stream_output_path != final_muxed_output_path:
                try: os.remove(video_stream_output_path)
                except Exception as e_clean: self.message_manager.add_warning(f"Could not remove temp upscaled video: {e_clean}")
            
            if final_output_path is None:
                 self.message_manager.add_error("Upscaling process did not determine a final output path.")
                 return None 

            progress(1.0, desc="Upscaling complete.")
            self.message_manager.add_success(f"Video upscaling complete: {final_output_path}")
            return final_output_path

        except Exception as e:
            self.message_manager.add_error(f"Error during video upscaling: {e}")
            import traceback
            self.message_manager.add_error(traceback.format_exc())
            progress(1.0, desc="Error during upscaling.")
            return None 
        
        finally:
            if reader: 
                try: 
                    if not reader.closed: 
                        reader.close()
                except Exception as e_close_finally: 
                    self.message_manager.add_warning(f"Error closing reader in finally: {e_close_finally}")
            
            if upscale_factor > 0 and self.esrgan_upscaler:
                self.esrgan_upscaler.unload_model(upscale_factor)
            
            devicetorch.empty_cache(torch)
            gc.collect()

    def tb_open_output_folder(self):
        
        folder_path = os.path.abspath(self.toolbox_permanent_save_dir)
        try:
            os.makedirs(folder_path, exist_ok=True) 
            if sys.platform == 'win32': subprocess.run(['explorer', folder_path])
            elif sys.platform == 'darwin': subprocess.run(['open', folder_path])
            else: subprocess.run(['xdg-open', folder_path])
            self.message_manager.add_success(f"Opened permanent save folder: {folder_path}")
        except Exception as e:
            self.message_manager.add_error(f"Error opening folder {folder_path}: {e}")

    def tb_clear_temporary_files(self):
        temp_dir_path_str = str(self._base_temp_output_dir)
        self.message_manager.add_message(f"Attempting to clear temporary files in: {temp_dir_path_str}", "INFO")
        
        file_count = 0
        dir_count = 0
        cleared_successfully = False

        if os.path.exists(temp_dir_path_str):
            try:
                # Optional: Count items before deleting for a more informative message
                for item in os.listdir(temp_dir_path_str):
                    item_path = os.path.join(temp_dir_path_str, item)
                    if os.path.isfile(item_path):
                        file_count += 1
                    elif os.path.isdir(item_path):
                        dir_count +=1
                
                shutil.rmtree(temp_dir_path_str)
                self.message_manager.add_success(
                    f"Successfully removed temporary directory and its contents ({file_count} files, {dir_count} subdirectories)."
                )
                cleared_successfully = True
            except Exception as e:
                self.message_manager.add_error(f"Error deleting temporary directory '{temp_dir_path_str}': {e}")
                self.message_manager.add_error(traceback.format_exc())
                cleared_successfully = False # Explicitly false
        else:
            self.message_manager.add_message("Temporary directory does not exist. Nothing to clear.", "INFO")
            cleared_successfully = True # Considered success as there's nothing to do

        # Always try to recreate the base temporary directory
        try:
            os.makedirs(temp_dir_path_str, exist_ok=True)
            if cleared_successfully: # Only add this if the primary clear operation wasn't an error
                 self.message_manager.add_message(f"Recreated temporary directory: {temp_dir_path_str}", "INFO")
        except Exception as e_recreate:
            self.message_manager.add_error(f"CRITICAL: Failed to recreate temporary directory '{temp_dir_path_str}': {e_recreate}. Processing may fail.")
            self.message_manager.add_error(traceback.format_exc())
            cleared_successfully = False # If recreate fails, it's not a full success

        return cleared_successfully    
