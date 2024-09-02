import os
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
import numpy as np

from diffusers import StableDiffusionPipeline

from layer_diffuse.models import TransparentVAEDecoder
from layer_diffuse.loaders import load_lora_to_unet
from layer_diffuse.utils import rgba2rgbfp32, crop_and_resize_image



if __name__ == "__main__":

    model_path = hf_hub_download(
        'LayerDiffusion/layerdiffusion-v1',
        'layer_sd15_vae_transparent_decoder.safetensors',
    )

    vae_transparent_decoder = TransparentVAEDecoder.from_pretrained("digiplay/Juggernaut_final", subfolder="vae", torch_dtype=torch.float16).to("cuda")
    vae_transparent_decoder.set_transparent_decoder(load_file(model_path), mod_number=2)

    pipeline = StableDiffusionPipeline.from_pretrained("digiplay/Juggernaut_final", vae=vae_transparent_decoder, torch_dtype=torch.float16, safety_checker=None).to("cuda")

    model_path = hf_hub_download(
        'LayerDiffusion/layerdiffusion-v1',
        'layer_sd15_bg2fg.safetensors'
    )

    kwargs_encoder = load_lora_to_unet(pipeline.unet, model_path, frames=2, use_control=True)

    bg_image = np.array(Image.open(os.path.join("assets", "bg_cond.png")))
    bg_image = crop_and_resize_image(rgba2rgbfp32(bg_image), 1, 512, 512)
    bg_image = torch.from_numpy(np.ascontiguousarray(bg_image[None].copy())).movedim(-1, 1)
    bg_image = bg_image.cpu().float() * 2.0 - 1.0
    bg_signal = kwargs_encoder(bg_image)


    
    image = pipeline(prompt="a dog sitting in room, high quality", 
                     width=512, height=512,
                     cross_attention_kwargs={"layerdiffuse_control_signals": bg_signal},
                     num_images_per_prompt=2, return_dict=False)[0]


    image[0].save("result.png")
    image[1].save("result1.png")

    