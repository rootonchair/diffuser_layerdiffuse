import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from layer_diffuse.loaders import (
    enable_unet_extra_concat_condition,
    merge_sdxl_concat_delta_weights_into_unet,
)
from layer_diffuse.utils import crop_and_resize_image, rgba2rgbfp32


DEFAULT_WEIGHT_REPO = "rootonchair/diffuser_layerdiffuse"
DEFAULT_WEIGHT_NAME = "diffuser_layer_xl_bg2ble.safetensors"


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
    parser = argparse.ArgumentParser(description="Generate an SDXL blended image from a background condition.")
    parser.add_argument(
        "--weight",
        default=DEFAULT_WEIGHT_NAME,
        help="Weight filename in --weight-repo. The file is loaded from the Hugging Face cache.",
    )
    parser.add_argument("--weight-repo", default=DEFAULT_WEIGHT_REPO, help="Hugging Face repo for remote weights.")
    parser.add_argument("--background", default="assets/bg_cond_forge_sanity.png")
    parser.add_argument("--output", default="result_xl_bg2ble.png")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--variant", default="fp16", help="Model variant to load. Use 'none' for repos without variants.")
    parser.add_argument("--no-use-safetensors", action="store_true", help="Allow loading Diffusers .bin component weights.")
    parser.add_argument("--vae", default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--prompt", default="a dog sitting in room, high quality")
    parser.add_argument("--negative-prompt", default="bad, ugly")
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cpu-offload", action="store_true", help="Use accelerate CPU offload instead of .to('cuda').")
    return parser.parse_args()


def download_weight(weight, repo_id):
    return hf_hub_download(repo_id=repo_id, filename=weight)


def make_generator(seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)


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
    # Forge's bg2ble sanity check uses DPM++ 2M SDE Karras for the blending pass.
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


if __name__ == "__main__":
    args = parse_args()
    weight_path = download_weight(args.weight, args.weight_repo)
    if not Path(args.background).exists():
        raise FileNotFoundError(f"Condition image not found: {args.background}")

    pipeline = load_pipeline(args)

    diff_state_dict = load_file(str(weight_path))
    merge_sdxl_concat_delta_weights_into_unet(pipeline, diff_state_dict)

    background_latents = encode_condition_image(pipeline, args.background, args.width, args.height)
    enable_unet_extra_concat_condition(pipeline.unet, background_latents)

    images = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=make_generator(args.seed),
        num_images_per_prompt=1,
        return_dict=False,
    )[0]
    images[0].save(args.output)
