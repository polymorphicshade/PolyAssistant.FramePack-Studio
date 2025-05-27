import threading
import time
import uuid
import json
import os
import zipfile
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import queue as queue_module  # Renamed to avoid conflicts
import io
import base64
from PIL import Image
import numpy as np

from diffusers_helper.thread_utils import AsyncStream
from modules.pipelines.metadata_utils import create_metadata
from modules.settings import Settings
from diffusers_helper.gradio.progress_bar import make_progress_bar_html


# Simple LIFO queue implementation to avoid dependency on queue.LifoQueue
class SimpleLifoQueue:
    def __init__(self):
        self._queue = []
        self._mutex = threading.Lock()
        self._not_empty = threading.Condition(self._mutex)
    
    def put(self, item):
        with self._mutex:
            self._queue.append(item)
            self._not_empty.notify()
    
    def get(self):
        with self._not_empty:
            while not self._queue:
                self._not_empty.wait()
            return self._queue.pop()
    
    def task_done(self):
        pass  # For compatibility with queue.Queue


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    params: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[str] = None
    progress_data: Optional[Dict] = None
    queue_position: Optional[int] = None
    stream: Optional[Any] = None
    input_image: Optional[np.ndarray] = None
    latent_type: Optional[str] = None
    thumbnail: Optional[str] = None
    generation_type: Optional[str] = None # Added generation_type
    input_image_saved: bool = False  # Flag to track if input image has been saved
    end_frame_image_saved: bool = False  # Flag to track if end frame image has been saved

    def __post_init__(self):
        # Store generation type
        self.generation_type = self.params.get('model_type', 'Original') # Initialize generation_type

        # Store input image or latent type
        if 'input_image' in self.params and self.params['input_image'] is not None:
            self.input_image = self.params['input_image']
            # Create thumbnail
            if isinstance(self.input_image, np.ndarray):
                # Handle numpy array (image)
                img = Image.fromarray(self.input_image)
                img.thumbnail((100, 100))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
            elif isinstance(self.input_image, str):
                # Handle string (video path)
                # Create a generic video thumbnail
                img = Image.new('RGB', (100, 100), (0, 0, 128))  # Blue for video
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
            else:
                # Handle other types
                self.thumbnail = None
        elif 'latent_type' in self.params:
            self.latent_type = self.params['latent_type']
            # Create a colored square based on latent type
            color_map = {
                "Black": (0, 0, 0),
                "White": (255, 255, 255),
                "Noise": (128, 128, 128),
                "Green Screen": (0, 177, 64)
            }
            color = color_map.get(self.latent_type, (0, 0, 0))
            img = Image.new('RGB', (100, 100), color)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"


class VideoJobQueue:
    def __init__(self):
        self.queue = queue_module.Queue()  # Using standard Queue instead of LifoQueue
        self.jobs = {}
        self.current_job = None
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.worker_function = None  # Will be set from outside
        self.is_processing = False  # Flag to track if we're currently processing a job
    
    def set_worker_function(self, worker_function):
        """Set the worker function to use for processing jobs"""
        self.worker_function = worker_function
    
    def serialize_job(self, job):
        """Serialize a job to a JSON-compatible format"""
        try:
            # Create a simplified representation of the job
            serialized = {
                "id": job.id,
                "status": job.status.value,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error": job.error,
                "result": job.result,
                "queue_position": job.queue_position,
                "generation_type": job.generation_type,
            }
            
            # Add simplified params (excluding complex objects)
            serialized_params = {}
            for k, v in job.params.items():
                if k not in ["input_image", "end_frame_image", "stream"]:
                    # Try to include only JSON-serializable values
                    try:
                        # Test if value is JSON serializable
                        json.dumps({k: v})
                        serialized_params[k] = v
                    except (TypeError, OverflowError):
                        # Skip non-serializable values
                        pass
            
            # Handle LoRA information specifically
            # Only include selected LoRAs for the generation
            if "selected_loras" in job.params and job.params["selected_loras"]:
                selected_loras = job.params["selected_loras"]
                # Ensure it's a list
                if not isinstance(selected_loras, list):
                    selected_loras = [selected_loras] if selected_loras is not None else []
                
                # Get LoRA values if available
                lora_values = job.params.get("lora_values", [])
                if not isinstance(lora_values, list):
                    lora_values = [lora_values] if lora_values is not None else []
                
                # Get loaded LoRA names
                lora_loaded_names = job.params.get("lora_loaded_names", [])
                if not isinstance(lora_loaded_names, list):
                    lora_loaded_names = [lora_loaded_names] if lora_loaded_names is not None else []
                
                # Create LoRA data dictionary
                lora_data = {}
                for lora_name in selected_loras:
                    try:
                        # Find the index of the LoRA in loaded names
                        idx = lora_loaded_names.index(lora_name) if lora_loaded_names else -1
                        # Get the weight value
                        weight = lora_values[idx] if lora_values and idx >= 0 and idx < len(lora_values) else 1.0
                        # Handle weight as list
                        if isinstance(weight, list):
                            weight_value = weight[0] if weight and len(weight) > 0 else 1.0
                        else:
                            weight_value = weight
                        # Store as float
                        lora_data[lora_name] = float(weight_value)
                    except (ValueError, IndexError):
                        # Default weight if not found
                        lora_data[lora_name] = 1.0
                    except Exception as e:
                        print(f"Error processing LoRA {lora_name}: {e}")
                        lora_data[lora_name] = 1.0
                
                # Add to serialized params
                serialized_params["loras"] = lora_data
            
            serialized["params"] = serialized_params
            
            # Don't include the thumbnail as it can be very large and cause issues
            # if job.thumbnail:
            #     serialized["thumbnail"] = job.thumbnail
                
            return serialized
        except Exception as e:
            print(f"Error serializing job {job.id}: {e}")
            # Return minimal information that should always be serializable
            return {
                "id": job.id,
                "status": job.status.value,
                "error": f"Error serializing: {str(e)}"
            }
    
    def save_queue_to_json(self):
        """Save the current queue to queue.json using the central metadata utility"""
        try:
            # Make a copy of job IDs to avoid holding the lock while serializing
            with self.lock:
                job_ids = list(self.jobs.keys())
            
            # Create a settings instance
            settings = Settings()
            
            # Create a directory to store queue images if it doesn't exist
            queue_images_dir = "queue_images"
            os.makedirs(queue_images_dir, exist_ok=True)
            
            # Serialize jobs outside the lock using metadata_utils
            serialized_jobs = {}
            for job_id in job_ids:
                job = self.get_job(job_id)
                if job:
                    # Try to use metadata_utils.create_metadata if possible
                    try:
                        # Create metadata using the central utility
                        metadata = create_metadata(job.params, job.id, settings.settings)
                        
                        # Add job status and other fields not included in metadata
                        metadata.update({
                            "id": job.id,
                            "status": job.status.value,
                            "created_at": job.created_at,
                            "started_at": job.started_at,
                            "completed_at": job.completed_at,
                            "error": job.error,
                            "result": job.result,
                            "queue_position": job.queue_position,
                        })
                        
                        # Only save images if the job is running or completed
                        if job.status in [JobStatus.RUNNING, JobStatus.COMPLETED]:
                            # Save input image to disk if it exists and hasn't been saved yet
                            if 'input_image' in job.params and isinstance(job.params['input_image'], np.ndarray) and not job.input_image_saved:
                                input_image_path = os.path.join(queue_images_dir, f"{job_id}_input.png")
                                try:
                                    Image.fromarray(job.params['input_image']).save(input_image_path)
                                    metadata["saved_input_image_path"] = input_image_path
                                    print(f"Saved input image for job {job_id} to {input_image_path}")
                                    # Mark the image as saved
                                    job.input_image_saved = True
                                except Exception as e:
                                    print(f"Error saving input image for job {job_id}: {e}")
                            elif 'input_image' in job.params and isinstance(job.params['input_image'], np.ndarray) and job.input_image_saved:
                                # If the image has already been saved, just add the path to metadata
                                input_image_path = os.path.join(queue_images_dir, f"{job_id}_input.png")
                                if os.path.exists(input_image_path):
                                    metadata["saved_input_image_path"] = input_image_path
                            
                            # Save end frame image to disk if it exists and hasn't been saved yet
                            if 'end_frame_image' in job.params and isinstance(job.params['end_frame_image'], np.ndarray) and not job.end_frame_image_saved:
                                end_frame_image_path = os.path.join(queue_images_dir, f"{job_id}_end_frame.png")
                                try:
                                    Image.fromarray(job.params['end_frame_image']).save(end_frame_image_path)
                                    metadata["saved_end_frame_image_path"] = end_frame_image_path
                                    print(f"Saved end frame image for job {job_id} to {end_frame_image_path}")
                                    # Mark the end frame image as saved
                                    job.end_frame_image_saved = True
                                except Exception as e:
                                    print(f"Error saving end frame image for job {job_id}: {e}")
                            elif 'end_frame_image' in job.params and isinstance(job.params['end_frame_image'], np.ndarray) and job.end_frame_image_saved:
                                # If the end frame image has already been saved, just add the path to metadata
                                end_frame_image_path = os.path.join(queue_images_dir, f"{job_id}_end_frame.png")
                                if os.path.exists(end_frame_image_path):
                                    metadata["saved_end_frame_image_path"] = end_frame_image_path
                        
                        serialized_jobs[job_id] = metadata
                    except Exception as e:
                        print(f"Error using metadata_utils for job {job_id}: {e}")
                        # Fall back to the old serialization method
                        serialized_jobs[job_id] = self.serialize_job(job)
            
            # Save to file
            with open("queue.json", "w") as f:
                json.dump(serialized_jobs, f, indent=2)
                
            print(f"Saved {len(serialized_jobs)} jobs to queue.json")
        except Exception as e:
            print(f"Error saving queue to JSON: {e}")
    
    def add_job(self, params):
        """Add a job to the queue and return its ID"""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            params=params,
            status=JobStatus.PENDING,
            created_at=time.time(),
            progress_data={},
            stream=AsyncStream(),
            input_image_saved=False,  # Initialize as not saved for new jobs
            end_frame_image_saved=False  # Initialize as not saved for new jobs
        )
        
        with self.lock:
            print(f"Adding job {job_id} to queue, current job is {self.current_job.id if self.current_job else 'None'}")
            self.jobs[job_id] = job
            self.queue.put(job_id)
        
        # Save the queue to JSON after adding a new job (outside the lock)
        try:
            self.save_queue_to_json()
        except Exception as e:
            print(f"Error saving queue to JSON after adding job: {e}")
        
        return job_id
    
    def get_job(self, job_id):
        """Get job by ID"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self):
        """Get all jobs"""
        with self.lock:
            return list(self.jobs.values())
    
    def cancel_job(self, job_id):
        """Cancel a pending job"""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
                
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()  # Mark completion time
                result = True
            elif job.status == JobStatus.RUNNING:
                # Send cancel signal to the job's stream
                if hasattr(job, 'stream') and job.stream:
                    job.stream.input_queue.push('end')
                    
                # Mark job as cancelled (this will be confirmed when the worker processes the end signal)
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()  # Mark completion time
                result = True
            else:
                result = False
        
        # Save the queue to JSON after cancelling a job (outside the lock)
        if result:
            try:
                self.save_queue_to_json()
            except Exception as e:
                print(f"Error saving queue to JSON after cancelling job: {e}")
        
        return result
    
    def clear_queue(self):
        """Cancel all pending jobs in the queue"""
        cancelled_count = 0
        try:
            # First, make a copy of all pending job IDs to avoid modifying the dictionary during iteration
            with self.lock:
                # Get all pending job IDs
                pending_job_ids = [job_id for job_id, job in self.jobs.items() 
                                if job.status == JobStatus.PENDING]
            
            # Cancel each pending job individually
            for job_id in pending_job_ids:
                try:
                    with self.lock:
                        job = self.jobs.get(job_id)
                        if job and job.status == JobStatus.PENDING:
                            job.status = JobStatus.CANCELLED
                            job.completed_at = time.time()
                            cancelled_count += 1
                except Exception as e:
                    print(f"Error cancelling job {job_id}: {e}")
            
            # Now clear the queue
            with self.lock:
                # Clear the queue (this doesn't affect running jobs)
                queue_items_cleared = 0
                try:
                    while not self.queue.empty():
                        try:
                            self.queue.get_nowait()
                            self.queue.task_done()
                            queue_items_cleared += 1
                        except queue_module.Empty:
                            break
                except Exception as e:
                    print(f"Error clearing queue: {e}")
            
            # Save the updated queue state
            try:
                self.save_queue_to_json()
            except Exception as e:
                print(f"Error saving queue state: {e}")
            
            print(f"Cleared {cancelled_count} jobs from the queue")
            return cancelled_count
        except Exception as e:
            import traceback
            print(f"Error in clear_queue: {e}")
            traceback.print_exc()
            return 0
            
    def clear_completed_jobs(self):
        """Remove cancelled or completed jobs from the queue"""
        removed_count = 0
        try:
            # First, make a copy of all completed/cancelled job IDs to avoid modifying the dictionary during iteration
            with self.lock:
                # Get all completed or cancelled job IDs
                completed_job_ids = [job_id for job_id, job in self.jobs.items() 
                                  if job.status in [JobStatus.COMPLETED, JobStatus.CANCELLED]]
            
            # Remove each completed/cancelled job individually
            for job_id in completed_job_ids:
                try:
                    with self.lock:
                        if job_id in self.jobs:
                            del self.jobs[job_id]
                            removed_count += 1
                except Exception as e:
                    print(f"Error removing job {job_id}: {e}")
            
            # Save the updated queue state
            try:
                self.save_queue_to_json()
            except Exception as e:
                print(f"Error saving queue state: {e}")
            
            print(f"Removed {removed_count} completed/cancelled jobs from the queue")
            return removed_count
        except Exception as e:
            import traceback
            print(f"Error in clear_completed_jobs: {e}")
            traceback.print_exc()
            return 0
    
    def get_queue_position(self, job_id):
        """Get position in queue (0 = currently running)"""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
                
            if job.status == JobStatus.RUNNING:
                return 0
                
            if job.status != JobStatus.PENDING:
                return None
                
            # Count pending jobs ahead in queue
            position = 1  # Start at 1 because 0 means running
            for j in self.jobs.values():
                if (j.status == JobStatus.PENDING and 
                    j.created_at < job.created_at):
                    position += 1
            return position
    
    def update_job_progress(self, job_id, progress_data):
        """Update job progress data"""
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.progress_data = progress_data
    
    def export_queue_to_zip(self, output_path=None):
        """Export the current queue to a zip file containing queue.json and queue_images directory
        
        Args:
            output_path: Path to save the zip file. If None, uses 'queue_export.zip' in the current directory.
            
        Returns:
            str: Path to the created zip file
        """
        try:
            # Use default path if none provided
            if output_path is None:
                output_path = "queue_export.zip"
            
            # Make sure queue.json is up to date
            self.save_queue_to_json()
            
            # Create a zip file
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add queue.json to the zip file
                if os.path.exists("queue.json"):
                    zipf.write("queue.json")
                    print(f"Added queue.json to {output_path}")
                else:
                    print("Warning: queue.json not found, creating an empty one")
                    with open("queue.json", "w") as f:
                        json.dump({}, f)
                    zipf.write("queue.json")
                
                # Add queue_images directory to the zip file if it exists
                queue_images_dir = "queue_images"
                if os.path.exists(queue_images_dir) and os.path.isdir(queue_images_dir):
                    for root, _, files in os.walk(queue_images_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Add file to zip with path relative to queue_images_dir
                            arcname = os.path.join(os.path.basename(queue_images_dir), file)
                            zipf.write(file_path, arcname)
                            print(f"Added {file_path} to {output_path}")
                else:
                    print(f"Warning: {queue_images_dir} directory not found or empty")
                    # Create the directory if it doesn't exist
                    os.makedirs(queue_images_dir, exist_ok=True)
            
            print(f"Queue exported to {output_path}")
            return output_path
            
        except Exception as e:
            import traceback
            print(f"Error exporting queue to zip: {e}")
            traceback.print_exc()
            return None
    
    def load_queue_from_json(self, file_path=None):
        """Load queue from a JSON file or zip file
        
        Args:
            file_path: Path to the JSON or ZIP file. If None, uses 'queue.json' in the current directory.
            
        Returns:
            int: Number of jobs loaded
        """
        try:
            # Use default path if none provided
            if file_path is None:
                file_path = "queue.json"
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Queue file not found: {file_path}")
                return 0
            
            # Check if it's a zip file
            if file_path.lower().endswith('.zip'):
                return self._load_queue_from_zip(file_path)
            
            # Load the JSON data
            with open(file_path, 'r') as f:
                serialized_jobs = json.load(f)
            
            # Count of jobs loaded
            loaded_count = 0
            
            # Process each job
            with self.lock:
                for job_id, job_data in serialized_jobs.items():
                    # Skip if job already exists
                    if job_id in self.jobs:
                        print(f"Job {job_id} already exists, skipping")
                        continue
                    
                    # Skip completed, failed, or cancelled jobs
                    status = job_data.get('status')
                    if status in ['completed', 'failed', 'cancelled']:
                        print(f"Skipping job {job_id} with status {status}")
                        continue
                    
                    # If the job was running when saved, we'll need to set it as the current job
                    was_running = (status == 'running')
                    
                    # Extract relevant fields to construct params
                    params = {
                        # Basic parameters
                        'model_type': job_data.get('model_type', 'Original'),
                        'prompt_text': job_data.get('prompt', ''),
                        'n_prompt': job_data.get('negative_prompt', ''),
                        'seed': job_data.get('seed', 0),
                        'steps': job_data.get('steps', 25),
                        'cfg': job_data.get('cfg', 1.0),
                        'gs': job_data.get('gs', 10.0),
                        'rs': job_data.get('rs', 0.0),
                        'latent_type': job_data.get('latent_type', 'Black'),
                        'total_second_length': job_data.get('total_second_length', 6),
                        'blend_sections': job_data.get('blend_sections', 4),
                        'latent_window_size': job_data.get('latent_window_size', 9),
                        'resolutionW': job_data.get('resolutionW', 640),
                        'resolutionH': job_data.get('resolutionH', 640),
                        
                        # Initialize image parameters
                        'input_image': None,
                        'end_frame_image': None,
                        'end_frame_strength': job_data.get('end_frame_strength', 1.0),
                        'use_teacache': job_data.get('use_teacache', True),
                        'teacache_num_steps': job_data.get('teacache_num_steps', 25),
                        'teacache_rel_l1_thresh': job_data.get('teacache_rel_l1_thresh', 0.15),
                        'has_input_image': job_data.get('has_input_image', True),
                    }
                    
                    # Load input image from disk if saved path exists
                    if "saved_input_image_path" in job_data and os.path.exists(job_data["saved_input_image_path"]):
                        try:
                            input_image_path = job_data["saved_input_image_path"]
                            print(f"Loading input image from {input_image_path}")
                            input_image = np.array(Image.open(input_image_path))
                            params['input_image'] = input_image
                            params['has_input_image'] = True
                        except Exception as e:
                            print(f"Error loading input image for job {job_id}: {e}")
                    
                    # Load end frame image from disk if saved path exists
                    if "saved_end_frame_image_path" in job_data and os.path.exists(job_data["saved_end_frame_image_path"]):
                        try:
                            end_frame_image_path = job_data["saved_end_frame_image_path"]
                            print(f"Loading end frame image from {end_frame_image_path}")
                            end_frame_image = np.array(Image.open(end_frame_image_path))
                            params['end_frame_image'] = end_frame_image
                        except Exception as e:
                            print(f"Error loading end frame image for job {job_id}: {e}")
                    
                    # Add LoRA information if present
                    if 'loras' in job_data:
                        lora_data = job_data.get('loras', {})
                        selected_loras = list(lora_data.keys())
                        lora_values = list(lora_data.values())
                        params['selected_loras'] = selected_loras
                        params['lora_values'] = lora_values
                    
                    # Get settings for output_dir and metadata_dir
                    settings = Settings()
                    output_dir = settings.get("output_dir")
                    metadata_dir = settings.get("metadata_dir")
                    input_files_dir = settings.get("input_files_dir")
                    
                    # Add these directories to the params
                    params['output_dir'] = output_dir
                    params['metadata_dir'] = metadata_dir
                    params['input_files_dir'] = input_files_dir
                    
                    # Create a new job
                    job = Job(
                        id=job_id,
                        params=params,
                        status=JobStatus(job_data.get('status', 'pending')),
                        created_at=job_data.get('created_at', time.time()),
                        progress_data={},
                        stream=AsyncStream(),
                        # Mark images as saved if their paths exist in the job data
                        input_image_saved="saved_input_image_path" in job_data and os.path.exists(job_data["saved_input_image_path"]),
                        end_frame_image_saved="saved_end_frame_image_path" in job_data and os.path.exists(job_data["saved_end_frame_image_path"])
                    )
                    
                    # Add job to the queue
                    self.jobs[job_id] = job
                    
                    # Only add pending jobs to the processing queue
                    if job.status == JobStatus.PENDING:
                        self.queue.put(job_id)
                        loaded_count += 1
                    # If the job was running, set it as the current job
                    elif was_running and self.current_job is None:
                        print(f"Setting job {job_id} as current job (was running when saved)")
                        self.current_job = job
                        # Create a new stream for the resumed job
                        job.stream = AsyncStream()
                        # Initialize progress_data if it doesn't exist
                        if not hasattr(job, 'progress_data') or job.progress_data is None:
                            job.progress_data = {}
                        # Mark it as running again
                        job.status = JobStatus.RUNNING
                        loaded_count += 1
            
            print(f"Loaded {loaded_count} pending jobs from {file_path}")
            return loaded_count
            
        except Exception as e:
            import traceback
            print(f"Error loading queue from JSON: {e}")
            traceback.print_exc()
            return 0
    
    def _load_queue_from_zip(self, zip_path):
        """Load queue from a zip file
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            int: Number of jobs loaded
        """
        try:
            # Create a temporary directory to extract the zip file
            temp_dir = "temp_queue_import"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Check if queue.json exists in the extracted files
            queue_json_path = os.path.join(temp_dir, "queue.json")
            if not os.path.exists(queue_json_path):
                print(f"queue.json not found in {zip_path}")
                shutil.rmtree(temp_dir)
                return 0
            
            # Check if queue_images directory exists in the extracted files
            queue_images_dir = os.path.join(temp_dir, "queue_images")
            if os.path.exists(queue_images_dir) and os.path.isdir(queue_images_dir):
                # Copy the queue_images directory to the current directory
                target_queue_images_dir = "queue_images"
                os.makedirs(target_queue_images_dir, exist_ok=True)
                
                # Copy all files from the extracted queue_images directory to the target directory
                for file in os.listdir(queue_images_dir):
                    src_path = os.path.join(queue_images_dir, file)
                    dst_path = os.path.join(target_queue_images_dir, file)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied {src_path} to {dst_path}")
            
            # Load the queue from the extracted queue.json
            loaded_count = self.load_queue_from_json(queue_json_path)
            
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)
            
            return loaded_count
            
        except Exception as e:
            import traceback
            print(f"Error loading queue from zip: {e}")
            traceback.print_exc()
            # Clean up the temporary directory if it exists
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return 0
    
    def _worker_loop(self):
        """Worker thread that processes jobs from the queue"""
        while True:
            try:
                # Get the next job ID from the queue
                try:
                    job_id = self.queue.get(block=True, timeout=1.0)  # Use timeout to allow periodic checks
                except queue_module.Empty:
                    # No jobs in queue, just continue the loop
                    continue
                
                with self.lock:
                    job = self.jobs.get(job_id)
                    if not job:
                        self.queue.task_done()
                        continue
                    
                    # Skip cancelled jobs
                    if job.status == JobStatus.CANCELLED:
                        self.queue.task_done()
                        continue
                    
                    # If we're already processing a job, wait for it to complete
                    if self.is_processing:
                        # Put the job back in the queue
                        self.queue.put(job_id)
                        self.queue.task_done()
                        time.sleep(0.1)  # Small delay to prevent busy waiting
                        continue
                    
                    # Check if there's a previously running job that was interrupted
                    previously_running_job = None
                    for j in self.jobs.values():
                        if j.status == JobStatus.RUNNING and j.id != job_id:
                            previously_running_job = j
                            break
                    
                    # If there's a previously running job, process it first
                    if previously_running_job:
                        print(f"Found previously running job {previously_running_job.id}, processing it first")
                        # Put the current job back in the queue
                        self.queue.put(job_id)
                        self.queue.task_done()
                        # Process the previously running job
                        job = previously_running_job
                        job_id = previously_running_job.id
                        
                        # Create a new stream for the resumed job and initialize progress_data
                        job.stream = AsyncStream()
                        job.progress_data = {}
                        
                        # Push an initial progress update to the stream
                        from diffusers_helper.gradio.progress_bar import make_progress_bar_html
                        job.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Resuming job...'))))
                    
                    print(f"Starting job {job_id}, current job was {self.current_job.id if self.current_job else 'None'}")
                    job.status = JobStatus.RUNNING
                    job.started_at = time.time()
                    self.current_job = job
                    self.is_processing = True
                
                job_completed = False
                
                try:
                    if self.worker_function is None:
                        raise ValueError("Worker function not set. Call set_worker_function() first.")
                    
                    # Start the worker function with the job parameters
                    from diffusers_helper.thread_utils import async_run
                    print(f"Starting worker function for job {job_id}")
                    async_run(
                        self.worker_function,
                        **job.params,
                        job_stream=job.stream
                    )
                    print(f"Worker function started for job {job_id}")
                    
                    # Process the results from the stream
                    output_filename = None
                    
                    # Track activity time for logging purposes
                    last_activity_time = time.time()
                    
                    while True:
                        # Check if job has been cancelled before processing next output
                        with self.lock:
                            if job.status == JobStatus.CANCELLED:
                                print(f"Job {job_id} was cancelled, breaking out of processing loop")
                                job_completed = True
                                break
                        
                        # Get current time for activity checks
                        current_time = time.time()
                        
                        # Check for inactivity (no output for a while)
                        if current_time - last_activity_time > 60:  # 1 minute of inactivity
                            print(f"Checking if job {job_id} is still active...")
                            # Just a periodic check, don't break yet
                        
                        try:
                            # Try to get data from the queue with a non-blocking approach
                            flag, data = job.stream.output_queue.next()
                            
                            # Update activity time since we got some data
                            last_activity_time = time.time()
                            
                            if flag == 'file':
                                output_filename = data
                                with self.lock:
                                    job.result = output_filename
                            
                            elif flag == 'progress':
                                preview, desc, html = data
                                with self.lock:
                                    job.progress_data = {
                                        'preview': preview,
                                        'desc': desc,
                                        'html': html
                                    }
                            
                            elif flag == 'end':
                                print(f"Received end signal for job {job_id}")
                                job_completed = True
                                break
                                
                        except IndexError:
                            # Queue is empty, wait a bit and try again
                            time.sleep(0.1)
                            continue
                        except Exception as e:
                            print(f"Error processing job output: {e}")
                            # Wait a bit before trying again
                            time.sleep(0.1)
                            continue
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error processing job {job_id}: {e}")
                    with self.lock:
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        job.completed_at = time.time()
                    job_completed = True
                
                finally:
                    with self.lock:
                        # Make sure we properly clean up the job state
                        if job.status == JobStatus.RUNNING:
                            if job_completed:
                                job.status = JobStatus.COMPLETED
                            else:
                                # Something went wrong but we didn't mark it as completed
                                job.status = JobStatus.FAILED
                                job.error = "Job processing was interrupted"
                            
                            job.completed_at = time.time()
                        
                        print(f"Finishing job {job_id} with status {job.status}")
                        self.is_processing = False
                        self.current_job = None
                        self.queue.task_done()
                    
                    # Save the queue to JSON after job completion (outside the lock)
                    try:
                        self.save_queue_to_json()
                    except Exception as e:
                        print(f"Error saving queue to JSON after job completion: {e}")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error in worker loop: {e}")
                
                # Make sure we reset processing state if there was an error
                with self.lock:
                    self.is_processing = False
                    if self.current_job:
                        self.current_job.status = JobStatus.FAILED
                        self.current_job.error = f"Worker loop error: {str(e)}"
                        self.current_job.completed_at = time.time()
                        self.current_job = None
                
                time.sleep(0.5)  # Prevent tight loop on error
