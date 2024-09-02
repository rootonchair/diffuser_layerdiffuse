from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch

from diffusers import StableDiffusionPipeline

from layer_diffuse.models import TransparentVAEDecoder
from layer_diffuse.loaders import load_lora_to_unet



if __name__ == "__main__":
    model_path = hf_hub_download(
        'LayerDiffusion/layerdiffusion-v1',
        'layer_sd15_vae_transparent_decoder.safetensors',
    )

    vae_transparent_decoder = TransparentVAEDecoder.from_pretrained("digiplay/Juggernaut_final", subfolder="vae", torch_dtype=torch.float16).to("cuda")
    vae_transparent_decoder.set_transparent_decoder(load_file(model_path), mod_number=3)
    pipeline = StableDiffusionPipeline.from_pretrained("digiplay/Juggernaut_final", vae=vae_transparent_decoder, safety_checker=None, torch_dtype=torch.float16).to("cuda")

    model_path = hf_hub_download(
        'LayerDiffusion/layerdiffusion-v1',
        'layer_sd15_joint.safetensors'
    )

    load_lora_to_unet(pipeline.unet, model_path, frames=3)
    
    image = pipeline(prompt="a dog sitting in room, high quality", width=512, height=512, num_images_per_prompt=3, return_dict=False)[0]


    image[0].save("result_joint_0.png")
    image[1].save("result_joint_1.png")
    image[2].save("result_joint_2.png")

    