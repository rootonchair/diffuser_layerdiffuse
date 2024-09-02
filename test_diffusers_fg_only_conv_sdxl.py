from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch

from diffusers import StableDiffusionXLPipeline

from layer_diffuse.models import TransparentVAEDecoder
from layer_diffuse.loaders import merge_delta_weights_into_unet


if __name__ == "__main__":

    transparent_vae = TransparentVAEDecoder.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    transparent_vae.config.force_upcast = False
    model_path = hf_hub_download(
        'LayerDiffusion/layerdiffusion-v1',
        'vae_transparent_decoder.safetensors',
    )
    transparent_vae.set_transparent_decoder(load_file(model_path))

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        vae=transparent_vae,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    model_path = hf_hub_download(
        'rootonchair/diffuser_layerdiffuse',
        'diffuser_layer_xl_transparent_conv.safetensors'
    )
    diff_state_dict = load_file(model_path)
    merge_delta_weights_into_unet(pipeline, diff_state_dict)

    
    seed = torch.randint(high=1000000, size=(1,)).item()
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    negative_prompt = "bad quality, distorted"
    images = pipeline(prompt=prompt, 
                       negative_prompt=negative_prompt,
                       generator=torch.Generator(device='cuda').manual_seed(seed),
                       num_images_per_prompt=1, return_dict=False)[0]

    images[0].save("result_conv_sdxl.png")
    