import torch.nn as nn
import torch
import cv2
import numpy as np
import importlib.metadata
from packaging.version import parse
from tqdm import tqdm
from typing import Optional, Tuple, Union
from PIL import Image

from diffusers import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput


diffusers_version = importlib.metadata.version('diffusers')

def check_diffusers_version(min_version="0.25.0"):
    assert parse(diffusers_version) >= parse(
        min_version
    ), f"diffusers>={min_version} requirement not satisfied. Please install correct diffusers version."

check_diffusers_version()

if parse(diffusers_version) >= parse("0.29.0"):
    from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
else:
    from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class LatentTransparencyOffsetEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            zero_module(torch.nn.Conv2d(256, 4, kernel_size=3, padding=1, stride=1)),
        )

    def __call__(self, x):
        return self.blocks(x)


# 1024 * 1024 * 3 -> 16 * 16 * 512 -> 1024 * 1024 * 3
class UNet1024(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (32, 32, 64, 128, 256, 512, 512),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        self.latent_conv_in = zero_module(nn.Conv2d(4, block_out_channels[2], kernel_size=1))

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift="default",
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=None,
            add_attention=True,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift="default",
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, latent):
        sample_latent = self.latent_conv_in(latent)
        sample = self.conv_in(x)
        emb = None

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if i == 3:
                sample = sample + sample_latent

            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb)

        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


class TransparentVAEDecoder(AutoencoderKL):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: float = True,
    ):
        self.mod_number = None
        super().__init__(in_channels, out_channels, down_block_types, up_block_types, block_out_channels, layers_per_block, act_fn, latent_channels, norm_num_groups, sample_size, scaling_factor, latents_mean, latents_std, force_upcast)

    def set_transparent_decoder(self, sd, mod_number=1):
        model = UNet1024(in_channels=3, out_channels=4)
        model.load_state_dict(sd, strict=True)
        model.to(device=self.device, dtype=self.dtype)
        model.eval()

        self.transparent_decoder = model
        self.mod_number = mod_number
    
    def estimate_single_pass(self, pixel, latent):
        y = self.transparent_decoder(pixel, latent)
        return y

    def estimate_augmented(self, pixel, latent):
        args = [
            [False, 0], [False, 1], [False, 2], [False, 3], [True, 0], [True, 1], [True, 2], [True, 3],
        ]

        result = []

        for flip, rok in tqdm(args):
            feed_pixel = pixel.clone()
            feed_latent = latent.clone()

            if flip:
                feed_pixel = torch.flip(feed_pixel, dims=(3,))
                feed_latent = torch.flip(feed_latent, dims=(3,))

            feed_pixel = torch.rot90(feed_pixel, k=rok, dims=(2, 3))
            feed_latent = torch.rot90(feed_latent, k=rok, dims=(2, 3))

            eps = self.estimate_single_pass(feed_pixel, feed_latent).clip(0, 1)
            eps = torch.rot90(eps, k=-rok, dims=(2, 3))

            if flip:
                eps = torch.flip(eps, dims=(3,))

            result += [eps]

        result = torch.stack(result, dim=0)
        median = torch.median(result, dim=0).values
        return median

    def decode(self, z: torch.Tensor, return_dict: bool = True, generator=None) -> Union[DecoderOutput, torch.Tensor]:
        pixel = super().decode(z, return_dict=False, generator=generator)[0]
        pixel = pixel / 2 + 0.5


        result_pixel = []
        for i in range(int(z.shape[0])):
            if self.mod_number is None or (self.mod_number != 1 and i % self.mod_number != 0):
                img = torch.cat((pixel[i:i+1], torch.ones_like(pixel[i:i+1,:1,:,:])), dim=1)
                result_pixel.append(img)
                continue

            y = self.estimate_augmented(pixel[i:i+1], z[i:i+1])

            y = y.clip(0, 1).movedim(1, -1)
            alpha = y[..., :1]
            fg = y[..., 1:]

            B, H, W, C = fg.shape
            cb = checkerboard(shape=(H // 64, W // 64))
            cb = cv2.resize(cb, (W, H), interpolation=cv2.INTER_NEAREST)
            cb = (0.5 + (cb - 0.5) * 0.1)[None, ..., None]
            cb = torch.from_numpy(cb).to(fg)

            png = torch.cat([fg, alpha], dim=3)
            png = png.permute(0, 3, 1, 2)
            result_pixel.append(png)
        
        result_pixel = torch.cat(result_pixel, dim=0)
        result_pixel = (result_pixel - 0.5) * 2

        if not return_dict:
            return (result_pixel, )
        return DecoderOutput(sample=result_pixel)


def build_alpha_pyramid(color, alpha, dk=1.2):
    pyramid = []
    current_premultiplied_color = color * alpha
    current_alpha = alpha

    while True:
        pyramid.append((current_premultiplied_color, current_alpha))

        H, W, C = current_alpha.shape
        if min(H, W) == 1:
            break

        current_premultiplied_color = cv2.resize(current_premultiplied_color, (int(W / dk), int(H / dk)), interpolation=cv2.INTER_AREA)
        current_alpha = cv2.resize(current_alpha, (int(W / dk), int(H / dk)), interpolation=cv2.INTER_AREA)[:, :, None]
    return pyramid[::-1]


def pad_rgb(np_rgba_hwc_uint8):
    np_rgba_hwc = np_rgba_hwc_uint8.astype(np.float32) / 255.0
    pyramid = build_alpha_pyramid(color=np_rgba_hwc[..., :3], alpha=np_rgba_hwc[..., 3:])

    top_c, top_a = pyramid[0]
    fg = np.sum(top_c, axis=(0, 1), keepdims=True) / np.sum(top_a, axis=(0, 1), keepdims=True).clip(1e-8, 1e32)

    for layer_c, layer_a in pyramid:
        layer_h, layer_w, _ = layer_c.shape
        fg = cv2.resize(fg, (layer_w, layer_h), interpolation=cv2.INTER_LINEAR)
        fg = layer_c + fg * (1.0 - layer_a)

    return fg


def convert_rgba2rgb(img):
    background = Image.new("RGB", img.size, (127, 127, 127))
    background.paste(img, mask=img.split()[3])
    return background


class TransparentVAEEncoder:
    def __init__(self, sd, device="cpu", torch_dtype=torch.float32):
        self.load_device = device
        self.dtype = torch_dtype

        model = LatentTransparencyOffsetEncoder()
        model.load_state_dict(sd, strict=True)
        model.to(device=self.load_device, dtype=self.dtype)
        model.eval()

        self.model = model
    
    @torch.no_grad()
    def _encode(self, image):
        list_of_np_rgba_hwc_uint8 = [np.array(image)]
        list_of_np_rgb_padded = [pad_rgb(x) for x in list_of_np_rgba_hwc_uint8]
        rgb_padded_bchw_01 = torch.from_numpy(np.stack(list_of_np_rgb_padded, axis=0)).float().movedim(-1, 1)
        rgba_bchw_01 = torch.from_numpy(np.stack(list_of_np_rgba_hwc_uint8, axis=0)).float().movedim(-1, 1) / 255.0
        a_bchw_01 = rgba_bchw_01[:, 3:, :, :]
        offset_feed = torch.cat([a_bchw_01, rgb_padded_bchw_01], dim=1).to(device=self.load_device, dtype=self.dtype)
        offset = self.model(offset_feed)
        return offset

    def encode(self, image, pipeline, mask=None):
        latent_offset = self._encode(image)

        init_image = convert_rgba2rgb(image)
        
        init_image = pipeline.image_processor.preprocess(init_image)
        init_image = init_image.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
        latents = pipeline.vae.encode(init_image).latent_dist
        latents = latents.mean + latents.std * latent_offset.to(latents.mean)
        latents = pipeline.vae.config.scaling_factor * latents

        if mask is not None:
            mask = pipeline.mask_processor.preprocess(mask)
            mask = mask.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
            masked_image = init_image * (mask < 0.5)
            masked_image_latents = pipeline.vae.encode(masked_image).latent_dist
            masked_image_latents = masked_image_latents.mean + masked_image_latents.std * latent_offset.to(masked_image_latents.mean)
            masked_image_latents = pipeline.vae.config.scaling_factor * masked_image_latents
            
            return latents, masked_image_latents

        return latents