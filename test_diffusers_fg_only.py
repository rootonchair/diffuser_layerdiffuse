import argparse

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from layer_diffuse.loaders import load_lora_to_unet
from layer_diffuse.models import TransparentVAEDecoder


DEFAULT_WEIGHT_REPO = "LayerDiffusion/layerdiffusion-v1"
DEFAULT_WEIGHT_NAME = "layer_sd15_transparent_attn.safetensors"
DEFAULT_TRANSPARENT_DECODER_NAME = "layer_sd15_vae_transparent_decoder.safetensors"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SD1.5 transparent foreground images.")
    parser.add_argument("--weight", default=DEFAULT_WEIGHT_NAME)
    parser.add_argument("--weight-repo", default=DEFAULT_WEIGHT_REPO)
    parser.add_argument("--transparent-decoder", default=DEFAULT_TRANSPARENT_DECODER_NAME)
    parser.add_argument("--model", default="digiplay/Juggernaut_final")
    parser.add_argument("--vae-subfolder", default="vae")
    parser.add_argument("--prompt", default="a dog sitting in room, high quality")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-images-per-prompt", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outputs", nargs="+", default=["result.png", "result1.png", "result2.png"])
    parser.add_argument("--cpu-offload", action="store_true", help="Use accelerate CPU offload instead of .to('cuda').")
    return parser.parse_args()


def download_weight(weight, repo_id):
    return hf_hub_download(repo_id=repo_id, filename=weight)


def make_generator(seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)


def load_pipeline(args):
    decoder_path = download_weight(args.transparent_decoder, args.weight_repo)
    transparent_vae = TransparentVAEDecoder.from_pretrained(
        args.model,
        subfolder=args.vae_subfolder,
        torch_dtype=torch.float16,
    )
    transparent_vae.set_transparent_decoder(load_file(str(decoder_path)))

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model,
        vae=transparent_vae,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cuda")
    return pipeline


if __name__ == "__main__":
    args = parse_args()
    weight_path = download_weight(args.weight, args.weight_repo)
    pipeline = load_pipeline(args)
    load_lora_to_unet(pipeline.unet, weight_path, frames=1)

    images = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        generator=make_generator(args.seed),
        num_images_per_prompt=args.num_images_per_prompt,
        return_dict=False,
    )[0]

    for image, output in zip(images, args.outputs):
        image.save(output)
