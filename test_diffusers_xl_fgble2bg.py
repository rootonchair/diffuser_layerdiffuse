import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from PIL import Image
from safetensors.torch import load_file

from layer_diffuse.loaders import (
    enable_unet_extra_concat_condition,
    merge_sdxl_concat_delta_weights_into_unet,
)
from layer_diffuse.utils import crop_and_resize_image, rgba2rgbfp32


def encode_condition_image(pipeline, image_path, width, height):
    image = Image.open(image_path).convert("RGBA")
    image = np.array(image)
    image = crop_and_resize_image(rgba2rgbfp32(image), 1, height, width)
    image = torch.from_numpy(np.ascontiguousarray(image[None].copy())).movedim(-1, 1)
    execution_device = getattr(pipeline, "_execution_device", pipeline.vae.device)
    image = image.to(device=execution_device, dtype=pipeline.vae.dtype)
    image = image * 2.0 - 1.0
    with torch.no_grad():
        latents = pipeline.vae.encode(image).latent_dist.mean
    return latents * pipeline.vae.config.scaling_factor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an SDXL background from foreground and blended-image conditions."
    )
    parser.add_argument("--weight", default="weights/diffuser_layer_xl_fgble2bg.safetensors")
    parser.add_argument("--foreground", default="assets/sdxl_fg_cond_detailed.png")
    parser.add_argument("--blend", default="assets/sdxl_fg2ble_detailed_default_scheduler.png")
    parser.add_argument("--output", default="result_xl_fgble2bg.png")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--variant", default="fp16", help="Model variant to load. Use 'none' for repos without variants.")
    parser.add_argument("--no-use-safetensors", action="store_true", help="Allow loading Diffusers .bin component weights.")
    parser.add_argument("--vae", default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--prompt", default="room, high quality")
    parser.add_argument("--negative-prompt", default="bad, ugly")
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument(
        "--ending-step",
        type=float,
        default=0.62,
        help="Fraction of denoising steps that use the LayerDiffuse fg+blend condition.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cpu-offload", action="store_true", help="Use accelerate CPU offload instead of .to('cuda').")
    return parser.parse_args()


def load_pipeline(args):
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float16)
    vae.config.force_upcast = False

    variant = None if args.variant.lower() in ("", "none", "null") else args.variant
    pipeline_kwargs = dict(
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=not args.no_use_safetensors,
        add_watermarker=False,
    )
    if variant is not None:
        pipeline_kwargs["variant"] = variant

    pipeline = StableDiffusionXLPipeline.from_pretrained(args.model, **pipeline_kwargs)
    # The fgble2bg weight is very scheduler-sensitive. DPM++ 2M SDE Karras
    # gave the cleanest background removal in parity tests with the Forge workflow.
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        solver_order=2,
        use_karras_sigmas=True,
    )
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cuda")
    return pipeline


def get_remaining_timesteps(pipeline, steps, ending_step):
    pipeline.scheduler.set_timesteps(steps)
    cutoff = int(
        round(
            pipeline.scheduler.config.num_train_timesteps
            - (ending_step * pipeline.scheduler.config.num_train_timesteps)
        )
    )
    return [int(t.item()) for t in pipeline.scheduler.timesteps if int(t.item()) < cutoff]


def snapshot_scheduler_state(scheduler):
    # The first pass stops before denoising is complete, so the base-UNet pass
    # must resume with DPM's multistep history instead of restarting the solver.
    def to_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, list):
            return [to_cpu(item) for item in value]
        return value

    return {
        "_step_index": scheduler._step_index,
        "lower_order_nums": scheduler.lower_order_nums,
        "model_outputs": to_cpu(scheduler.model_outputs),
    }


def restore_scheduler_state(scheduler, state, device):
    def to_device(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, list):
            return [to_device(item) for item in value]
        return value

    scheduler._step_index = state["_step_index"]
    scheduler.lower_order_nums = state["lower_order_nums"]
    scheduler.model_outputs = to_device(state["model_outputs"])


def make_generator(seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)


def run_pipeline(pipeline, args, **kwargs):
    return pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance_scale,
        generator=make_generator(args.seed),
        num_images_per_prompt=1,
        return_dict=False,
        **kwargs,
    )[0]


@torch.no_grad()
def continue_base_pipeline(pipeline, args, latents, remaining_timesteps, scheduler_state):
    # Forge switches back to the original UNet after `ending_step`. Diffusers
    # does not expose that handoff directly, so we continue the remaining
    # timesteps manually with the saved scheduler state and the base UNet.
    device = pipeline._execution_device
    pipeline._guidance_scale = args.guidance_scale
    pipeline._guidance_rescale = 0.0
    pipeline._clip_skip = None
    pipeline._cross_attention_kwargs = None
    pipeline.scheduler.set_timesteps(args.steps, device=device)
    restore_scheduler_state(pipeline.scheduler, scheduler_state, device)

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=args.prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
        negative_prompt=args.negative_prompt,
    )

    original_size = (args.height, args.width)
    target_size = (args.height, args.width)
    add_text_embeds = pooled_prompt_embeds
    text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim
    add_time_ids = pipeline._get_add_time_ids(
        original_size,
        (0, 0),
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    if pipeline.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(1, 1)
    latents = latents.to(device=device, dtype=prompt_embeds.dtype)
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(make_generator(args.seed), 0.0)

    remaining_timesteps = torch.tensor(remaining_timesteps, device=device, dtype=pipeline.scheduler.timesteps.dtype)
    with pipeline.progress_bar(total=len(remaining_timesteps)) as progress_bar:
        for timestep in remaining_timesteps:
            latent_model_input = (
                torch.cat([latents] * 2) if pipeline.do_classifier_free_guidance else latents
            )
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timestep)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = pipeline.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if pipeline.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipeline.scheduler.step(
                noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False
            )[0]
            progress_bar.update()

    latents = latents / pipeline.vae.config.scaling_factor
    image = pipeline.vae.decode(latents.to(dtype=pipeline.vae.dtype), return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type="pil")


if __name__ == "__main__":
    args = parse_args()
    weight_path = Path(args.weight)
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Converted weight not found at {weight_path}. Run "
            "`python scripts/convert_xl_fgble2bg.py --input /path/to/layer_xl_fgble2bg.safetensors` first."
        )
    for image_path in [args.foreground, args.blend]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Condition image not found: {image_path}")

    pipeline = load_pipeline(args)
    remaining_timesteps = get_remaining_timesteps(pipeline, args.steps, args.ending_step)
    diff_state_dict = load_file(str(weight_path))
    merge_sdxl_concat_delta_weights_into_unet(pipeline, diff_state_dict, extra_channels=8)
    foreground_latents = encode_condition_image(pipeline, args.foreground, args.width, args.height)
    blend_latents = encode_condition_image(pipeline, args.blend, args.width, args.height)
    if foreground_latents.shape != blend_latents.shape:
        raise ValueError(
            f"Foreground and blend latent shapes must match, got "
            f"{tuple(foreground_latents.shape)} and {tuple(blend_latents.shape)}."
        )
    enable_unet_extra_concat_condition(pipeline.unet, torch.cat([foreground_latents, blend_latents], dim=1))

    latents = run_pipeline(
        pipeline,
        args,
        num_inference_steps=args.steps,
        denoising_end=args.ending_step,
        output_type="latent",
    )
    scheduler_state = snapshot_scheduler_state(pipeline.scheduler)
    latents = latents.detach().cpu()

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pipeline = load_pipeline(args)
    images = continue_base_pipeline(pipeline, args, latents, remaining_timesteps, scheduler_state)
    images[0].save(args.output)
