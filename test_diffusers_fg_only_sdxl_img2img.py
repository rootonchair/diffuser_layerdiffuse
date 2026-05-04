import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLInpaintPipeline
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from layer_diffuse.models import TransparentVAEDecoder, TransparentVAEEncoder


DEFAULT_WEIGHT_REPO = "rootonchair/diffuser_layerdiffuse"
DEFAULT_WEIGHT_NAME = "diffuser_layer_xl_transparent_attn.safetensors"
DEFAULT_LAYERDIFFUSE_REPO = "LayerDiffusion/layerdiffusion-v1"
DEFAULT_TRANSPARENT_ENCODER_NAME = "vae_transparent_encoder.safetensors"
DEFAULT_TRANSPARENT_DECODER_NAME = "vae_transparent_decoder.safetensors"


def parse_args():
    parser = argparse.ArgumentParser(description="Inpaint an SDXL transparent foreground image.")
    parser.add_argument("--weight", default=DEFAULT_WEIGHT_NAME)
    parser.add_argument("--weight-repo", default=DEFAULT_WEIGHT_REPO)
    parser.add_argument("--layerdiffuse-repo", default=DEFAULT_LAYERDIFFUSE_REPO)
    parser.add_argument("--transparent-encoder", default=DEFAULT_TRANSPARENT_ENCODER_NAME)
    parser.add_argument("--transparent-decoder", default=DEFAULT_TRANSPARENT_DECODER_NAME)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--variant", default="fp16", help="Model variant to load. Use 'none' for repos without variants.")
    parser.add_argument("--no-use-safetensors", action="store_true", help="Allow loading Diffusers .bin component weights.")
    parser.add_argument("--vae", default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--init-image", default="assets/man_crop.png")
    parser.add_argument("--mask-image", default="assets/man_mask.png")
    parser.add_argument("--prompt", default="a handsome man")
    parser.add_argument("--negative-prompt", default="bad, ugly")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--output", default="result_inpaint_sdxl.png")
    parser.add_argument("--num-images-per-prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-offload", action="store_true", help="Use accelerate CPU offload instead of .to('cuda').")
    return parser.parse_args()


def download_weight(weight, repo_id):
    return hf_hub_download(repo_id=repo_id, filename=weight)


def make_generator(seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)


def load_pipeline(args):
    transparent_vae = TransparentVAEDecoder.from_pretrained(args.vae, torch_dtype=torch.float16)
    transparent_vae.config.force_upcast = False
    decoder_path = download_weight(args.transparent_decoder, args.layerdiffuse_repo)
    transparent_vae.set_transparent_decoder(load_file(str(decoder_path)))

    variant = None if args.variant.lower() in ("", "none", "null") else args.variant
    pipeline_kwargs = dict(
        vae=transparent_vae,
        torch_dtype=torch.float16,
        use_safetensors=not args.no_use_safetensors,
        add_watermarker=False,
    )
    if variant is not None:
        pipeline_kwargs["variant"] = variant

    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(args.model, **pipeline_kwargs)
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cuda")
    return pipeline


if __name__ == "__main__":
    args = parse_args()
    for image_path in [args.init_image, args.mask_image]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

    encoder_path = download_weight(args.transparent_encoder, args.layerdiffuse_repo)
    vae_transparent_encoder = TransparentVAEEncoder(load_file(str(encoder_path)))
    pipeline = load_pipeline(args)
    pipeline.load_lora_weights(args.weight_repo, weight_name=args.weight)

    init_image = Image.open(args.init_image).convert("RGBA").resize((args.width, args.height))
    mask_image = Image.open(args.mask_image).convert("L").resize((args.width, args.height))
    latents, masked_image_latents = vae_transparent_encoder.encode(init_image, pipeline, mask=mask_image)

    images = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=latents,
        masked_image_latents=masked_image_latents,
        strength=args.strength,
        mask_image=mask_image,
        generator=make_generator(args.seed),
        num_images_per_prompt=args.num_images_per_prompt,
        return_dict=False,
    )[0]
    images[0].save(args.output)
