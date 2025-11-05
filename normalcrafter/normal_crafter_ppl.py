from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import math

from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipelineOutput, StableVideoDiffusionPipeline
from PIL import Image
import cv2

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class NormalCrafterPipeline(StableVideoDiffusionPipeline):

    def _encode_image(self, image: torch.Tensor, device, num_videos_per_prompt, do_classifier_free_guidance, scale=1, image_size=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if isinstance(image, torch.Tensor):
            # Expected: (T,H,W,C) or (T,C,H,W) in [0,1]
            if image.dim() != 4:
                raise ValueError("Tensor image must be 4D (T,H,W,C) or (T,C,H,W).")
            if image.shape[-1] == 3:        # (T,H,W,C) -> (T,C,H,W)
                image = image.permute(0,3,1,2)
            elif image.shape[1] == 3:       # (T,C,H,W)
                pass
            else:
                raise ValueError("Channel dim must be 3.")

            pixel_values = image.to(device=device, dtype=torch.float32)  # B=T
            B, C, H, W = pixel_values.shape

            patches = [pixel_values]
            for i in range(1, scale):
                num_patches = i + 1
                ph = H // num_patches + 1
                pw = W // num_patches + 1
                for j in range(num_patches):
                    for k in range(num_patches):
                        patches.append(pixel_values[:, :, j*ph:(j+1)*ph, k*pw:(k+1)*pw])

            def encode_image(img_bchw):
                img_bchw = img_bchw * 2.0 - 1.0
                img_bchw = _resize_with_antialiasing(img_bchw, image_size or (224, 224))
                img_bchw = (img_bchw + 1.0) / 2.0  # zurück in [0,1]

                # CLIP-Preproc
                feat = self.feature_extractor(
                    images=img_bchw, do_normalize=True, do_center_crop=False,
                    do_resize=False, do_rescale=False, return_tensors="pt",
                ).pixel_values.to(device=device, dtype=dtype)

                emb = self.image_encoder(feat).image_embeds
                if emb.dim() < 3:
                    emb = emb.unsqueeze(1)
                return emb

            image_embeddings = torch.cat([encode_image(p) for p in patches], dim=1)
        else:
            raise ValueError("_encode_image() didn't get handed a torch-tensor")
            
        # Duplicate
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative, image_embeddings])

        return image_embeddings


    def encode_video_vae(self, images: torch.Tensor, chunk_size: int = 14):
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        device = self._execution_device

        if isinstance(images, torch.Tensor):
            # (T,H,W,C) oder (T,C,H,W) in [0,1] -> (1, C, T, H, W) in [-1,1]
            if images.shape[-1] == 3:        # (T,H,W,C) -> (T,C,H,W)
                imgs = images.permute(0,3,1,2)
            else:                             # (T,C,H,W)
                imgs = images
            imgs = imgs.to(device=device, dtype=self.vae.dtype)
            imgs = imgs * 2.0 - 1.0
            imgs = imgs.permute(1,0,2,3).unsqueeze(0)  # (C,T,H,W) -> (1,C,T,H,W)
        else:
            raise ValueError("encode_video_vae() didn't get handed a torch-tensor") 

        # (1,C,T,H,W) -> (C,T,H,W) -> (T,C,H,W)
        imgs = imgs.squeeze(0).permute(1,0,2,3)

        video_latents = []
        for i in range(0, imgs.shape[0], chunk_size):
            video_latents.append(self.vae.encode(imgs[i:i+chunk_size]).latent_dist.mode())
        image_latents = torch.cat(video_latents, dim=0)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        return image_latents
    
    @torch.no_grad()
    def __call__(
        self,
        images: torch.FloatTensor,
        decode_chunk_size: Optional[int] = None,
        time_step_size: Optional[int] = 1,
        window_size: Optional[int] = 1,
        fps: int = 7,  # <<< CHANGED: Added default, will be overridden by node
        motion_bucket_id: int = 127,  # <<< CHANGED: Added default, will be overridden by node
        noise_aug_strength: float = 0.0,  # <<< CHANGED: Added default, will be overridden by node
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True
    ):
        #images, pad_HWs = self.pad_image(images)

        # 0. Default height and width to unet
        if isinstance(images, torch.Tensor):
            if images.dim() != 4:
                raise ValueError("images tensor must be 4D (T,H,W,C) or (T,C,H,W)")
            if images.shape[-1] == 3:   # (T,H,W,C)
                T, H, W, C = images.shape
                images_for_enc = images.permute(0,3,1,2)  # (T,C,H,W)
            elif images.shape[1] == 3:  # (T,C,H,W)
                T, C, H, W = images.shape[0], images.shape[1], images.shape[2], images.shape[3]
                images_for_enc = images
            else:
                raise ValueError("Channel dim must be 3.")
            num_frames, height, width = T, H, W
        else:
            width, height = images[0].size
            num_frames = len(images)
            images_for_enc = images  # bleibt Liste von PIL

        # 1. Check inputs. Raise error if not correct -> Redundant, due to the rescaling done in normal_crafter_node.py
        # self.check_inputs(images, height, width) 

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = 1.0 # NormalCrafter seems to operate without CFG for normals directly
        num_videos_per_prompt = 1
        do_classifier_free_guidance = False # NormalCrafter specific
        num_inference_steps = 1 # For direct normal generation, this is typically 1 step if not iterative refinement
        # fps, motion_bucket_id, noise_aug_strength are now passed as arguments

        output_type = "pt" # Default output type from SVD pipeline is numpy array, but we use torch because we are cooler
        # data_keys = ["normal"] # Not used directly in this simplified flow
        use_linear_merge = True
        determineTrain = True # This seems to indicate a direct generation rather than denoising from pure noise
        encode_image_scale = 1
        encode_image_WH = None

        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else 7 # Default if not provided

        # 3. Encode input image using using clip. (num_image * num_videos_per_prompt, 1, 1024)
        image_embeddings = self._encode_image(images, device, num_videos_per_prompt, do_classifier_free_guidance=do_classifier_free_guidance, scale=encode_image_scale, image_size=encode_image_WH)
        # 4. Encode input image using VAE
        image_latents = self.encode_video_vae(images, chunk_size=decode_chunk_size).to(image_embeddings.dtype)

        # image_latents [num_frames, channels, height, width] ->[1, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(0)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,  # <<< USES PASSED-IN VALUE
            motion_bucket_id,  # <<< USES PASSED-IN VALUE
            noise_aug_strength,  # <<< USES PASSED-IN VALUE
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # get Start and End frame idx for each window
        def get_ses(num_frames_local): # Renamed to avoid conflict with outer num_frames
            ses = []
            for i in range(0, num_frames_local, time_step_size):
                ses.append([i, i + window_size])
            num_to_remain = 0
            for se_idx, se_val in enumerate(ses): # Use enumerate for clarity
                if se_val[1] > num_frames_local:
                    # Adjust the last window to not exceed num_frames_local if it's shorter than window_size
                    if se_idx > 0 and ses[se_idx-1][1] < num_frames_local : # Ensure there's a previous valid window
                        ses[se_idx] = [num_frames_local - window_size, num_frames_local]
                        if ses[se_idx][0] < 0: # Handle very short videos
                             ses[se_idx][0] = 0
                        num_to_remain = se_idx +1
                    else: # If this is the only window and it's too short or calculation leads to invalid range
                        ses[se_idx] = [0, num_frames_local] # Process what's available
                        num_to_remain = se_idx + 1

                    break # Stop adding more windows
                num_to_remain += 1
            ses = ses[:num_to_remain]

            # Ensure the last window covers the end if not already
            if ses and ses[-1][1] < num_frames_local and num_frames_local >= window_size :
                new_start = num_frames_local - window_size
                if new_start > ses[-1][0]: # Add only if it's a new, valid window
                    ses.append([new_start, num_frames_local])
                elif new_start < ses[-1][0] and new_start >= 0: # If it overlaps but starts earlier, replace last
                    ses[-1] = [new_start, num_frames_local]


            # Handle case where num_frames_local < window_size
            if not ses and num_frames_local > 0:
                ses.append([0, num_frames_local]) # Process all available frames

            return ses

        ses = get_ses(num_frames) # Use the original num_frames from the input 'images'

        pred = None
        for i, se in enumerate(ses):
            # Adjust window_num_frames based on the actual slice, esp. for the last potentially shorter window
            current_window_num_frames = se[1] - se[0]
            window_image_embeddings = image_embeddings[se[0]:se[1]]
            window_image_latents = image_latents[:, se[0]:se[1]]
            # added_time_ids might need adjustment if its batch size depends on num_frames,
            # but SVD usually computes it once for the whole sequence properties (fps, motion_bucket_id)
            window_added_time_ids = added_time_ids # Assuming added_time_ids are constant per generation call

            if i == 0 or time_step_size >= window_size: # Corrected condition: no overlap if step >= window
                to_replace_latents = None
            else:
                last_se = ses[i-1]
                overlap_start_in_current_window = 0
                overlap_start_in_previous_pred = last_se[1] - (window_size - time_step_size) # More robust overlap calculation
                num_to_replace_latents = window_size - time_step_size # This is the overlap size

                if num_to_replace_latents > 0 and pred is not None and pred.shape[1] >= (se[0] + num_to_replace_latents) :
                     # The slice from `pred` needs to correspond to the overlapping part of the *previous* window's output
                     # that aligns with the *start* of the current window's processing.
                     # `se[0]` is the global start index of the current window.
                     # `last_se[1]` is the global end index of the previous window.
                     # The overlap is from `se[0]` to `last_se[1]`.
                     # So, from `pred`, we need from `se[0]` up to `num_to_replace_latents` frames into the current window.
                    to_replace_latents = pred[:, se[0] : se[0] + num_to_replace_latents]
                else:
                    to_replace_latents = None


            latents_current_window = self.generate(
                num_inference_steps,
                device,
                batch_size,
                num_videos_per_prompt,
                current_window_num_frames, # Use actual frames in this window
                height, # Original height/width after padding
                width,  # Original height/width after padding
                window_image_embeddings,
                generator,
                determineTrain,
                to_replace_latents, # This is for conditioning the start of the current window
                do_classifier_free_guidance,
                window_image_latents,
                window_added_time_ids
            )

            # Merge latents
            if pred is None:
                pred = latents_current_window
            else:
                if to_replace_latents is not None and use_linear_merge and time_step_size < window_size:
                    num_overlap_frames = window_size - time_step_size
                    weight = torch.linspace(1.0, 0.0, num_overlap_frames + 2, device=device, dtype=latents_current_window.dtype)[1:-1]
                    weight = weight.view(1, -1, 1, 1, 1) # Reshape for broadcasting

                    # The part of `pred` to merge with is from global index `se[0]` for `num_overlap_frames`
                    pred_overlap_part = pred[:, se[0] : se[0] + num_overlap_frames]
                    # The part of `latents_current_window` to merge is its beginning
                    current_window_overlap_part = latents_current_window[:, :num_overlap_frames]

                    merged_overlap = pred_overlap_part * weight + current_window_overlap_part * (1.0 - weight)

                    # Update `pred` and append the new, non-overlapping part of `latents_current_window`
                    new_part_of_current_window = latents_current_window[:, num_overlap_frames:]
                    pred = torch.cat([pred[:, :se[0]], merged_overlap, new_part_of_current_window], dim=1)
                else:
                    # No overlap or no merge, just append the new part
                    # This assumes pred ends exactly where the new window (minus overlap) begins
                    pred = torch.cat([pred[:, :se[0] + time_step_size], latents_current_window[:, window_size - time_step_size:] ], dim=1)
                    # A simpler, but potentially less robust way if windows aren't perfectly aligned:
                    # pred = torch.cat([pred[:, :se[0]], latents_current_window], dim=1)
                    # This needs careful handling if just appending parts to avoid duplicated or missing frames.
                    # For now, let's assume `pred` is built up to `se[0]` and we append `latents_current_window`
                    # This logic needs to be very precise for stitching.
                    # A safer approach if `time_step_size` is not less than `window_size`:
                    if time_step_size >= window_size :
                        pred = torch.cat([pred, latents_current_window], dim=1) # Non-overlapping windows
                    else: # Overlapping windows but not merging (or linear merge disabled) - take new window's data
                         pred = torch.cat([pred[:, :se[0]], latents_current_window], dim=1)



        # Ensure pred is not longer than num_frames due to windowing logic
        if pred is not None and pred.shape[1] > num_frames:
            pred = pred[:, :num_frames]


        if not output_type == "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

            def decode_latents(latents_to_decode, num_frames_to_decode, decode_chunk_size_local):
                # Ensure latents_to_decode is not None and has frames
                if latents_to_decode is None or latents_to_decode.shape[1] == 0:
                    # Return a dummy tensor or raise an error, based on expected behavior
                    # For now, let's assume if latents are None, frames should be too.
                    # This part of SVD pipeline expects latents to be (1, T, C, H, W)
                    # If num_frames_to_decode is 0, it might also cause issues.
                    print("Warning: No latents to decode.")
                    # Create a dummy black frame sequence matching expected output structure
                    # This depends on what `self.video_processor.postprocess_video` expects.
                    # Assuming it handles (T, H, W, C) numpy arrays.
                    # Placeholder: return empty array or array of zeros.
                    # The original SVD decode_latents doesn't explicitly handle None input.
                    # Let's try to return a zero array matching expected shape if num_frames_to_decode > 0
                    if num_frames_to_decode > 0:
                        dummy_h = latents_to_decode.shape[3] * self.vae_scale_factor if latents_to_decode is not None else 256 # example H
                        dummy_w = latents_to_decode.shape[4] * self.vae_scale_factor if latents_to_decode is not None else 256 # example W
                        return np.zeros((num_frames_to_decode, dummy_h, dummy_w, 3), dtype=np.float32)
                    return np.array([])


                frames_decoded = self.decode_latents(latents_to_decode, num_frames_to_decode, decode_chunk_size_local)
                frames_decoded = self.video_processor.postprocess_video(video=frames_decoded, output_type=output_type)  # "pt"
                frames_decoded = frames_decoded * 2 - 1
                return frames_decoded

            frames = decode_latents(pred, num_frames, decode_chunk_size)
            
        else:
            frames = pred

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


    def generate(
            self, 
            num_inference_steps,
            device,
            batch_size,
            num_videos_per_prompt,
            num_frames,
            height,
            width,
            image_embeddings,
            generator,
            determineTrain,
            to_replace_latents,
            do_classifier_free_guidance,
            image_latents,
            added_time_ids,
            latents=None,
        ):
        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        if determineTrain:
            latents[...] = 0.

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # replace part of latents with conditons. ToDo: t embedding should also replace
                if to_replace_latents is not None:
                    num_img_condition = to_replace_latents.shape[1]
                    if not determineTrain:
                        _noise = randn_tensor(to_replace_latents.shape, generator=generator, device=device, dtype=image_embeddings.dtype)
                        noisy_to_replace_latents = self.scheduler.add_noise(to_replace_latents, _noise, t.unsqueeze(0))
                        latents[:, :num_img_condition] = noisy_to_replace_latents
                    else:
                        latents[:, :num_img_condition] = to_replace_latents
                    

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                timestep = t
                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_output = self.scheduler.step(noise_pred, t, latents)
                latents = scheduler_output.prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        return latents 
# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out
