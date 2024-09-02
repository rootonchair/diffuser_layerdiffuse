import os
from PIL import Image
from huggingface_hub import hf_hub_download
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

    pipeline = StableDiffusionPipeline.from_pretrained("digiplay/Juggernaut_final", vae=vae_transparent_decoder, torch_dtype=torch.float16, safety_checker=None).to("cuda")

    model_path = hf_hub_download(
        'LayerDiffusion/layerdiffusion-v1',
        'layer_sd15_fg2bg.safetensors'
    )

    kwargs_encoder = load_lora_to_unet(pipeline.unet, model_path, frames=2, use_control=True)

    fg_image = np.array(Image.open(os.path.join("assets", "fg_cond.png")))
    fg_image = crop_and_resize_image(rgba2rgbfp32(fg_image), 1, 512, 512)
    fg_image = torch.from_numpy(np.ascontiguousarray(fg_image[None].copy())).movedim(-1, 1)
    fg_image = fg_image.cpu().float() * 2.0 - 1.0
    fg_signal = kwargs_encoder(fg_image)


    
    image = pipeline(prompt="in room, high quality, 4K", 
                     width=512, height=512,
                     cross_attention_kwargs={"layerdiffuse_control_signals": fg_signal},
                     num_images_per_prompt=2, return_dict=False)[0]


    image[0].save("fg_result.png")
    image[1].save("fg_result1.png")

    