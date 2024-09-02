from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
from PIL import Image

from diffusers import StableDiffusionXLInpaintPipeline

from layer_diffuse.models import TransparentVAEDecoder, TransparentVAEEncoder


if __name__ == "__main__":
    model_path = hf_hub_download(
        'LayerDiffusion/layerdiffusion-v1',
        'vae_transparent_encoder.safetensors'
    )

    vae_transparent_encoder = TransparentVAEEncoder(load_file(model_path))

    transparent_vae = TransparentVAEDecoder.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
    transparent_vae.config.force_upcast = False
    model_path = hf_hub_download(
        'LayerDiffusion/layerdiffusion-v1',
        'vae_transparent_decoder.safetensors',
    )
    transparent_vae.set_transparent_decoder(load_file(model_path))

    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        vae=transparent_vae,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipeline.load_lora_weights('rootonchair/diffuser_layerdiffuse', weight_name='diffuser_layer_xl_transparent_attn.safetensors')

    init_image = Image.open("assets/man_crop.png").resize((1024, 1024))
    mask_image = Image.open("assets/man_mask.png")

    latents, masked_image_latents = vae_transparent_encoder.encode(init_image, pipeline, mask=mask_image)


    seed = 42 
    prompt = "a handsome man"
    negative_prompt = "bad, ugly"
    images = pipeline(prompt=prompt, 
                      negative_prompt=negative_prompt,
                      image=latents,
                      masked_image_latents=masked_image_latents,
                      strength=1.0,
                      mask_image=mask_image,
                      generator=torch.Generator(device='cuda').manual_seed(seed),
                      num_images_per_prompt=1, return_dict=False)[0]

    images[0].save("result_inpaint_sdxl.png")

    