# ComfyUI/custom_nodes/ComfyUI-NormalCrafter/normal_crafter_nodes.py

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import logging

import comfy.model_management
import model_management
import comfy.utils
from comfy.utils import common_upscale
import folder_paths # ComfyUI's way to get model paths

# Handle PIL version compatibility
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

# Try to import required components and provide guidance if they are missing
try:
    from diffusers import AutoencoderKLTemporalDecoder
    from huggingface_hub import snapshot_download
except ImportError:
    print("ComfyUI-NormalCrafter: Missing essential libraries. Please ensure 'diffusers', 'transformers', 'accelerate', and 'huggingface_hub' are installed.")
    # Consider raising an error or making the node unusable if critical dependencies are missing.

try:
    # This assumes the 'normalcrafter' directory is correctly placed within ComfyUI-NormalCrafter
    from .normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
    from .normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
except ImportError as e:
    print(f"ComfyUI-NormalCrafter: Error importing NormalCrafter components: {e}. "
          "Ensure the 'normalcrafter' directory is correctly placed inside 'ComfyUI-NormalCrafter'.")

# Global variable to cache the pipeline
NORMALCRAFTER_PIPE = None
CURRENT_PIPE_CONFIG = {} # Stores the config {"device": "cuda"/"cpu", "dtype": "float16"/"float32"}

# Define model paths and repo ID
NORMALCRAFTER_REPO_ID = "Yanrui95/NormalCrafter"
NORMALCRAFTER_MODELS_SUBDIR_NAME = "normalcrafter_models" # Subdirectory in ComfyUI/models/
SVD_XT_REPO_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

class DetailTransfer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target": ("IMAGE", ),
                "source": ("IMAGE", ),
                "mode": ([
                    "add",
                    "multiply",
                    "screen",
                    "overlay",
                    "soft_light",
                    "hard_light",
                    "color_dodge",
                    "color_burn",
                    "difference",
                    "exclusion",
                    "divide",
                    
                    ], 
                    {"default": "add"}
                    ),
                "blur_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01}),
                "blend_factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,  "round": 0.001}),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "NormalCrafter"

    def adjust_mask(self, mask, target_tensor):
        # Add a channel dimension and repeat to match the channel number of the target tensor
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)  # Add a channel dimension
            target_channels = target_tensor.shape[1]
            mask = mask.expand(-1, target_channels, -1, -1)  # Expand the channel dimension to match the target tensor's channels
    
        return mask


    def process(self, target, source, mode, blur_sigma, blend_factor, mask=None):
        B, H, W, C = target.shape
        device = model_management.get_torch_device()
        target_tensor = target.permute(0, 3, 1, 2).clone().to(device)
        source_tensor = source.permute(0, 3, 1, 2).clone().to(device)

        if target.shape[1:] != source.shape[1:]:
            source_tensor = comfy.utils.common_upscale(source_tensor, W, H, "bilinear", "disabled")

        if source.shape[0] < B:
            source = source[0].unsqueeze(0).repeat(B, 1, 1, 1)

        kernel_size = int(6 * int(blur_sigma) + 1)

        gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma))

        blurred_target = gaussian_blur(target_tensor)
        blurred_source = gaussian_blur(source_tensor)
        
        if mode == "add":
            tensor_out = (source_tensor - blurred_source) + blurred_target
        elif mode == "multiply":
            tensor_out = source_tensor * blurred_target
        elif mode == "screen":
            tensor_out = 1 - (1 - source_tensor) * (1 - blurred_target)
        elif mode == "overlay":
            tensor_out = torch.where(blurred_target < 0.5, 2 * source_tensor * blurred_target, 1 - 2 * (1 - source_tensor) * (1 - blurred_target))
        elif mode == "soft_light":
            tensor_out = (1 - 2 * blurred_target) * source_tensor**2 + 2 * blurred_target * source_tensor
        elif mode == "hard_light":
            tensor_out = torch.where(source_tensor < 0.5, 2 * source_tensor * blurred_target, 1 - 2 * (1 - source_tensor) * (1 - blurred_target))
        elif mode == "difference":
            tensor_out = torch.abs(blurred_target - source_tensor)
        elif mode == "exclusion":
            tensor_out = 0.5 - 2 * (blurred_target - 0.5) * (source_tensor - 0.5)
        elif mode == "color_dodge":
            tensor_out = blurred_target / (1 - source_tensor)
        elif mode == "color_burn":
            tensor_out = 1 - (1 - blurred_target) / source_tensor
        elif mode == "divide":
            tensor_out = (source_tensor / blurred_source) * blurred_target
        else:
            tensor_out = source_tensor
        
        tensor_out = torch.lerp(target_tensor, tensor_out, blend_factor)
        if mask is not None:
            # Call the function and pass in mask and target_tensor
            mask = self.adjust_mask(mask, target_tensor)
            mask = mask.to(device)
            tensor_out = torch.lerp(target_tensor, tensor_out, mask)
        tensor_out = torch.clamp(tensor_out, 0, 1)
        tensor_out = tensor_out.permute(0, 2, 3, 1).cpu().float()
        return (tensor_out,)

class NormalCrafterNode:
    def __init__(self):
        self.pipe = None # Instance variable for the pipeline
        self.last_processed_dimensions = None # Track last processed dimensions

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "max_res_dimension": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "window_size": ("INT", {"default": 14, "min": 2, "max": 64}),
                "time_step_size": ("INT", {"default": 10, "min": 1, "max": 64}),
                "decode_chunk_size": ("INT", {"default": 4, "min": 1, "max": 64}),
                "offload_pipe_to_cpu_on_finish": ("BOOLEAN", {"default": True}), # <<< NEW INPUT
                "use_xformers": (["auto", "disable", "force"], {"default": "auto"}), # <<< NEW INPUT
            },
            "optional": {
                "pipe_override": ("NORMALCRAFTER_PIPE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "NormalCrafter"

    def _get_local_nc_model_path(self):
        base_models_dir = os.path.join(folder_paths.models_dir, NORMALCRAFTER_MODELS_SUBDIR_NAME)
        os.makedirs(base_models_dir, exist_ok=True)
        model_name_folder = NORMALCRAFTER_REPO_ID.split('/')[-1]
        specific_model_path = os.path.join(base_models_dir, model_name_folder)
        return specific_model_path

    def _download_normalcrafter_model_if_needed(self):
        local_nc_path = self._get_local_nc_model_path()
        unet_config_path = os.path.join(local_nc_path, "unet", "config.json")

        if not os.path.exists(unet_config_path):
            print(f"ComfyUI-NormalCrafter: Downloading {NORMALCRAFTER_REPO_ID} model to {local_nc_path}...")
            try:
                snapshot_download(
                    repo_id=NORMALCRAFTER_REPO_ID,
                    local_dir=local_nc_path,
                    local_dir_use_symlinks=False,
                )
                print(f"ComfyUI-NormalCrafter: Model {NORMALCRAFTER_REPO_ID} download complete.")
            except Exception as e:
                print(f"ComfyUI-NormalCrafter: Failed to download model {NORMALCRAFTER_REPO_ID}: {e}")
                raise
        else:
            # print(f"ComfyUI-NormalCrafter: Model {NORMALCRAFTER_REPO_ID} found at {local_nc_path}.")
            pass
        return local_nc_path

    def _configure_xformers(self, pipe, use_xformers="auto"):
        """
        Configures xformers memory efficient attention for a given pipeline.
        Handles 'auto', 'disable', and 'force' modes.
        """
        if use_xformers == "disable":
            if hasattr(pipe, 'disable_xformers_memory_efficient_attention'):
                print("ComfyUI-NormalCrafter: Xformers memory efficient attention disabled.")
                try:
                    pipe.disable_xformers_memory_efficient_attention()
                except Exception as e:
                    print(f"ComfyUI-NormalCrafter: Warning - could not disable xformers: {e}")
            else:
                print("ComfyUI-NormalCrafter: Pipeline does not support disabling xformers.")
        elif use_xformers == "force":
            if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                print("ComfyUI-NormalCrafter: Xformers memory efficient attention forced on.")
                pipe.enable_xformers_memory_efficient_attention()
            else:
                print("ComfyUI-NormalCrafter: Warning - Pipeline does not support xformers, but 'force' was requested.")
        else: # "auto" or any other value
            if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    print("ComfyUI-NormalCrafter: Xformers memory efficient attention enabled (auto).")
                except Exception as e:
                    print(f"ComfyUI-NormalCrafter: Failed to enable Xformers memory efficient attention: {e}. Using standard attention.")
                    # If xformers fails, we can still use standard attention
                    pass
            else:
                print("ComfyUI-NormalCrafter: Pipeline does not support xformers. Using standard attention.")

    def _load_pipeline(self, device_str_requested="cuda", dtype_str_requested="float16", use_xformers="auto"):
        global NORMALCRAFTER_PIPE, CURRENT_PIPE_CONFIG

        target_device_for_this_run = comfy.model_management.get_torch_device() if device_str_requested == "cuda" else torch.device("cpu")
        
        # Determine the torch_dtype we want for this run
        if dtype_str_requested == "float16": final_torch_dtype_for_load = torch.float16
        elif dtype_str_requested == "bf16": final_torch_dtype_for_load = torch.bfloat16
        else: final_torch_dtype_for_load = torch.float32

        if NORMALCRAFTER_PIPE is not None:
            # Pipe exists. Check if its current dtype and target device are suitable.
            pipe_actual_dtype = NORMALCRAFTER_PIPE.dtype # The true dtype of the existing pipe
            pipe_actual_dtype_str = "float16" if pipe_actual_dtype == torch.float16 else \
                                    "bf16" if pipe_actual_dtype == torch.bfloat16 else "float32"

            if dtype_str_requested == pipe_actual_dtype_str:
                # Dtypes match. Just check device.
                if NORMALCRAFTER_PIPE.device != target_device_for_this_run:
                    print(f"ComfyUI-NormalCrafter: Moving existing pipeline from {NORMALCRAFTER_PIPE.device} to {target_device_for_this_run}.")
                    NORMALCRAFTER_PIPE.to(target_device_for_this_run) 
                    # The warning about fp16 on cpu will appear here if target_device_for_this_run is cpu, 
                    # but we're moving an fp16 pipe from cpu to gpu in the typical reload case, which is fine.
                
                CURRENT_PIPE_CONFIG = {"device": str(target_device_for_this_run), "dtype": dtype_str_requested}
                self.pipe = NORMALCRAFTER_PIPE
                # Configure xformers for existing pipe
                self._configure_xformers(NORMALCRAFTER_PIPE, use_xformers)
                # print(f"ComfyUI-NormalCrafter: Reusing existing pipeline. Now on {target_device_for_this_run} with {dtype_str_requested}.")
                return self.pipe
            else:
                # Dtype mismatch (e.g., user wants float32 now, but pipe is float16). Must reload fully.
                print(f"ComfyUI-NormalCrafter: Requested dtype {dtype_str_requested} differs from existing pipe's dtype {pipe_actual_dtype_str}. Reloading pipeline.")
                NORMALCRAFTER_PIPE = None # Force a full reload by clearing the global pipe
        
        # If NORMALCRAFTER_PIPE is None (either initially, or forced by dtype mismatch above)
        print("ComfyUI-NormalCrafter: Loading/Re-loading NormalCrafter pipeline from scratch...")
        local_nc_model_path = self._download_normalcrafter_model_if_needed()

        # Load components with final_torch_dtype_for_load
        unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
            local_nc_model_path, subfolder="unet", low_cpu_mem_usage=True, torch_dtype=final_torch_dtype_for_load
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            local_nc_model_path, subfolder="vae", low_cpu_mem_usage=True, torch_dtype=final_torch_dtype_for_load
        )
        svd_xt_variant = "fp16" if final_torch_dtype_for_load == torch.float16 else None

        pipe = NormalCrafterPipeline.from_pretrained(
            SVD_XT_REPO_ID, unet=unet, vae=vae, torch_dtype=final_torch_dtype_for_load, variant=svd_xt_variant,
        )

        # Configure xformers with our robust handler
        self._configure_xformers(pipe, use_xformers)

        pipe.to(target_device_for_this_run)
        NORMALCRAFTER_PIPE = pipe
        CURRENT_PIPE_CONFIG = {
            "device": str(target_device_for_this_run),
            "dtype": dtype_str_requested # The dtype it's configured with for this run
        }
        self.pipe = NORMALCRAFTER_PIPE
        print(f"ComfyUI-NormalCrafter: Pipeline loaded to {target_device_for_this_run} with dtype {final_torch_dtype_for_load}.")
        return self.pipe

    
    def make_safe_size(self, images: torch.Tensor, max_res_dim: int, device: torch.device) -> torch.Tensor:
        """
        Skaliert ein Batch von Bildern mit Torch, ohne PIL/NumPy,
        und sorgt gleich für Attention-kompatible Dimensionen.
        """
        imgs = images.to(device, non_blocking=True).permute(0, 3, 1, 2)  # (B,C,H,W)
        _, _, H, W = imgs.shape

        # 1. Scale down to max_res_dim (if necessary
        if max(H, W) > max_res_dim:
            scale = max_res_dim / max(H, W)
            target_h, target_w = int(round(H * scale)), int(round(W * scale))
        else:
            target_h, target_w = H, W

        # 2. Enforce Attention-Safe-Dimension (minimum 64)
        min_dim = 64
        align = 64
        target_h = max(min_dim, ((target_h + align - 1) // align) * align)
        target_w = max(min_dim, ((target_w + align - 1) // align) * align)

        # 3. Torch-Interpolation
        imgs_resized = F.interpolate(imgs, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return imgs_resized.permute(0, 2, 3, 1).contiguous()

    
    def process(self, images: torch.Tensor, seed: int, max_res_dimension: int,
                window_size: int, time_step_size: int, decode_chunk_size: int,
                offload_pipe_to_cpu_on_finish: bool, # <<< NEW PARAMETER
                use_xformers: str = "auto", # <<< NEW PARAMETER
                pipe_override=None):

        default_fps_for_time_ids = 7
        default_motion_bucket_id = 127
        default_noise_aug_strength = 0.0

        if pipe_override is not None:
            self.pipe = pipe_override
            print("ComfyUI-NormalCrafter: Using provided pipe_override.")
            
        else:
            current_comfy_device = comfy.model_management.get_torch_device()
            device_str = "cuda" if current_comfy_device.type == 'cuda' else "cpu"
            # Determine dtype based on device (float16 for CUDA, float32 for CPU)
            dtype_str = "float16" if device_str == "cuda" and comfy.model_management.should_use_fp16() else "float32"
            self._load_pipeline(device_str, dtype_str, use_xformers)

        if self.pipe is None:
            raise RuntimeError("ComfyUI-NormalCrafter: Pipeline could not be loaded.")
        
        # Ensure the pipe instance self.pipe is on the correct device for processing *before* using it
        # This is important if self.pipe came from the global cache and might have been on CPU
        processing_device = comfy.model_management.get_torch_device()
        if self.pipe.device != processing_device:
            print(f"ComfyUI-NormalCrafter: Moving self.pipe from {self.pipe.device} to {processing_device} for processing.")
            self.pipe.to(processing_device)

        #Resizing
        orig_h, orig_w = images.shape[1:3]
        resized_tensor = self.make_safe_size(images, max_res_dimension, processing_device)
        
        orig_h, orig_w = images.shape[1:3]
        resized_tensor = self.make_safe_size(images, max_res_dimension, processing_device)

        # Get number of frames in the tensor
        num_actual_frames = resized_tensor.shape[0]
 


        if num_actual_frames == 0:
             print("ComfyUI-NormalCrafter: Warning - No frames to process after resizing.")
             return (torch.zeros_like(images),)


        if num_actual_frames < window_size:
            print(f"ComfyUI-NormalCrafter: Number of frames ({num_actual_frames}) is less than window_size ({window_size}). Padding...")
            padding_needed = window_size - num_actual_frames
            last_frame = resized_tensor[-1:]  # Shape (1,H,W,C)
            pad_frames = last_frame.repeat(padding_needed, 1, 1, 1)
            effective_frames_for_pipeline = torch.cat([resized_tensor, pad_frames], dim=0)
        else:
            effective_frames_for_pipeline = resized_tensor

        
        
        generator_device = self.pipe.device # Should be processing_device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        effective_frame_count = effective_frames_for_pipeline.shape[0]

        print(f"ComfyUI-NormalCrafter: Processing {effective_frame_count} frames (effective) with seed {seed}. Original: {num_actual_frames}.")
        print(f"ComfyUI-NormalCrafter: Using (internal defaults) fps={default_fps_for_time_ids}, motion_id={default_motion_bucket_id}, noise_aug={default_noise_aug_strength}")

        pbar = comfy.utils.ProgressBar(effective_frame_count)  # optional, Pipe nutzt eh ihre eigene



        output_frames_pt = None
        error_occurred = False
        
        # Try processing with current settings
        try:
            with torch.inference_mode():
                output_frames_pt = self.pipe( # self.pipe should be on processing_device here
                    images=resized_tensor,
                    decode_chunk_size=decode_chunk_size,
                    time_step_size=time_step_size,
                    window_size=window_size,
                    fps=default_fps_for_time_ids,
                    motion_bucket_id=default_motion_bucket_id,
                    noise_aug_strength=default_noise_aug_strength,
                    generator=generator
                    # SVD pipeline has its own progress bar, no need to pass pbar here
                ).frames[0]
        except RuntimeError as e:
            if "CUDA error" in str(e) and "invalid configuration argument" in str(e) and use_xformers != "disable":
                print(f"ComfyUI-NormalCrafter: Flash attention error encountered: {str(e)}")
                error_occurred = True
                
                # Try disabling xformers and retrying
                if hasattr(self.pipe, 'disable_xformers_memory_efficient_attention'):
                    print("ComfyUI-NormalCrafter: Attempting to disable xformers and retry...")
                    self.pipe.disable_xformers_memory_efficient_attention()
                    
                    try:
                        with torch.inference_mode():
                            output_frames_pt = self.pipe(
                                images=resized_tensor,
                                decode_chunk_size=decode_chunk_size,
                                time_step_size=time_step_size,
                                window_size=window_size,
                                fps=default_fps_for_time_ids,
                                motion_bucket_id=default_motion_bucket_id,
                                noise_aug_strength=default_noise_aug_strength,
                                generator=generator
                            ).frames[0]
                        print("ComfyUI-NormalCrafter: Successfully processed without xformers.")
                    except Exception as retry_error:
                        print(f"ComfyUI-NormalCrafter: Retry without xformers also failed: {retry_error}")
                        raise retry_error
                    
                    # Re-enable xformers for future runs if it was auto mode
                    if use_xformers == "auto":
                        try:
                            self.pipe.enable_xformers_memory_efficient_attention()
                        except:
                            pass
                else:
                    raise e
            else:
                raise e
        
        if output_frames_pt is None:
            raise RuntimeError("ComfyUI-NormalCrafter: Failed to generate output frames.")

        if len(resized_tensor) > num_actual_frames:
            output_frames_pt = output_frames_pt[:num_actual_frames, ...]


        # Output from NormalCrafter has the pattern: [B,C,H,W]
        # 1. Normalize to a range of [0,1]
        output_frames_pt = (output_frames_pt.clamp(-1., 1.) * 0.5) + 0.5

        # 2. Resize back to input-resolution
        output_frames_pt = F.interpolate(
            output_frames_pt,
            size=(orig_h, orig_w),   # Eingangsauflösung
            mode="bilinear",
            align_corners=False
        )

        # 3. Permute Tensor to [B,H,W,C] for ComfyUI-Compatibility
        output_frames_pt = output_frames_pt.permute(0, 2, 3, 1).contiguous()



        # --- Explicit Offload After Processing ---
        # Only offload the globally managed pipe (NORMALCRAFTER_PIPE), not an overridden one.
        # And only if no pipe_override was used for this run.
        if pipe_override is None and \
           offload_pipe_to_cpu_on_finish and \
           NORMALCRAFTER_PIPE is not None and \
           NORMALCRAFTER_PIPE.device.type == 'cuda': # Check if it's on CUDA before moving
            print("ComfyUI-NormalCrafter: Offloading globally cached pipeline to CPU after processing.")
            try:
                NORMALCRAFTER_PIPE.to("cpu")
                global CURRENT_PIPE_CONFIG # Make sure to get the global
                if "device" in CURRENT_PIPE_CONFIG:
                    CURRENT_PIPE_CONFIG["device"] = "cpu" # Update its state to "cpu"
                else: # Should not happen if config is always set
                    CURRENT_PIPE_CONFIG = {"device": "cpu", "dtype": CURRENT_PIPE_CONFIG.get("dtype", "float32")}

                comfy.model_management.soft_empty_cache() # Ask ComfyUI to try and free VRAM
            except Exception as e:
                print(f"ComfyUI-NormalCrafter: Error offloading pipeline to CPU: {e}")
        
        print(f"ComfyUI-NormalCrafter: Processing complete. Output tensor shape: {output_frames_pt.shape}")
        return (output_frames_pt,)
