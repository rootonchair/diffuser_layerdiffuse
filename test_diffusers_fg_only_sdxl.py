import argparse

import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from layer_diffuse.models import TransparentVAEDecoder


DEFAULT_WEIGHT_REPO = "rootonchair/diffuser_layerdiffuse"
DEFAULT_WEIGHT_NAME = "diffuser_layer_xl_transparent_attn.safetensors"
DEFAULT_TRANSPARENT_DECODER_REPO = "LayerDiffusion/layerdiffusion-v1"
DEFAULT_TRANSPARENT_DECODER_NAME = "vae_transparent_decoder.safetensors"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SDXL transparent foreground images with attention injection.")
    parser.add_argument("--weight", default=DEFAULT_WEIGHT_NAME)
    parser.add_argument("--weight-repo", default=DEFAULT_WEIGHT_REPO)
    parser.add_argument("--transparent-decoder", default=DEFAULT_TRANSPARENT_DECODER_NAME)
    parser.add_argument("--transparent-decoder-repo", default=DEFAULT_TRANSPARENT_DECODER_REPO)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--variant", default="fp16", help="Model variant to load. Use 'none' for repos without variants.")
    parser.add_argument("--no-use-safetensors", action="store_true", help="Allow loading Diffusers .bin component weights.")
    parser.add_argument("--vae", default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--prompt", default="a cute corgi")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="result_sdxl.png")
    parser.add_argument("--num-images-per-prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
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
    decoder_path = download_weight(args.transparent_decoder, args.transparent_decoder_repo)
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

    pipeline = StableDiffusionXLPipeline.from_pretrained(args.model, **pipeline_kwargs)
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cuda")
    return pipeline


if __name__ == "__main__":
    args = parse_args()
    pipeline = load_pipeline(args)
    pipeline.load_lora_weights(args.weight_repo, weight_name=args.weight)

    images = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        generator=make_generator(args.seed),
        num_images_per_prompt=args.num_images_per_prompt,
        return_dict=False,
    )[0]
    images[0].save(args.output)
