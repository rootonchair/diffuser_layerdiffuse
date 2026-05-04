from safetensors.torch import load_file
import types
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, AttnProcessor2_0
from .models import LoraLoader, AttentionSharingProcessor, IPAdapterAttnShareProcessor, AttentionSharingProcessor2_0, IPAdapterAttnShareProcessor2_0


def merge_delta_weights_into_unet(pipe, delta_weights):
    unet_weights = pipe.unet.state_dict()

    for k in delta_weights.keys():
        assert k in unet_weights.keys(), k
        
    for key in delta_weights.keys():
        dtype = unet_weights[key].dtype
        unet_weights[key] = unet_weights[key].to(dtype=delta_weights[key].dtype) + delta_weights[key].to(device=unet_weights[key].device)
        unet_weights[key] = unet_weights[key].to(dtype)
    pipe.unet.load_state_dict(unet_weights, strict=True)
    return pipe


def _expand_unet_conv_in(unet, in_channels):
    old_conv = unet.conv_in
    if old_conv.in_channels == in_channels:
        return
    if in_channels < old_conv.in_channels:
        raise ValueError(
            f"Cannot shrink UNet conv_in from {old_conv.in_channels} to {in_channels} channels."
        )

    new_conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, : old_conv.in_channels].copy_(old_conv.weight)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    unet.conv_in = new_conv


def merge_sdxl_concat_delta_weights_into_unet(pipe, delta_weights, extra_channels=4):
    if "conv_in.weight" not in delta_weights:
        raise ValueError("Expected converted SDXL concat delta to include conv_in.weight.")

    conv_in_weight = delta_weights["conv_in.weight"]
    if conv_in_weight.ndim != 4:
        raise ValueError(
            f"Expected conv_in.weight to be a 4D convolution weight, got shape {tuple(conv_in_weight.shape)}."
        )
    target_channels = conv_in_weight.shape[1]
    base_channels = target_channels - extra_channels
    if pipe.unet.conv_in.in_channels == base_channels:
        _expand_unet_conv_in(pipe.unet, target_channels)
    elif pipe.unet.conv_in.in_channels != target_channels:
        raise ValueError(
            f"UNet conv_in has {pipe.unet.conv_in.in_channels} input channels, but the delta expects "
            f"{target_channels} channels ({base_channels} base + {extra_channels} condition)."
        )

    unet_weights = pipe.unet.state_dict()

    for key, delta in delta_weights.items():
        if key not in unet_weights:
            raise KeyError(key)
        if tuple(unet_weights[key].shape) != tuple(delta.shape):
            raise ValueError(
                f"Shape mismatch for {key}: UNet has {tuple(unet_weights[key].shape)}, "
                f"delta has {tuple(delta.shape)}."
            )

    for key, delta in delta_weights.items():
        dtype = unet_weights[key].dtype
        unet_weights[key] = (
            unet_weights[key].to(dtype=delta.dtype) + delta.to(device=unet_weights[key].device)
        ).to(dtype)
    pipe.unet.load_state_dict(unet_weights, strict=True)
    return pipe


def _match_extra_concat_condition(condition_latents, sample):
    condition_latents = condition_latents.to(device=sample.device, dtype=sample.dtype)
    if condition_latents.ndim != sample.ndim:
        raise ValueError(
            f"Condition latents must have {sample.ndim} dimensions, got {condition_latents.ndim}."
        )

    if condition_latents.shape[-2:] != sample.shape[-2:]:
        condition_latents = F.interpolate(
            condition_latents,
            size=sample.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    if condition_latents.shape[0] == sample.shape[0]:
        return condition_latents
    if condition_latents.shape[0] == 1:
        return condition_latents.repeat(sample.shape[0], 1, 1, 1)
    if sample.shape[0] == condition_latents.shape[0] * 2:
        return condition_latents.repeat(2, 1, 1, 1)
    raise ValueError(
        f"Condition batch {condition_latents.shape[0]} must match sample batch {sample.shape[0]}, "
        f"be 1 for a shared condition, or be half the sample batch for classifier-free guidance."
    )


def _timestep_key(timestep):
    if torch.is_tensor(timestep):
        timestep = timestep.flatten()[0].detach().float().item()
    return round(float(timestep), 6)


def enable_unet_extra_concat_condition(unet, condition_latents, base_unet=None, active_timesteps=None):
    if not hasattr(unet, "_layerdiffuse_original_forward"):
        unet._layerdiffuse_original_forward = unet.forward

        def forward_with_extra_concat(self, sample, timestep, *args, **kwargs):
            if (
                hasattr(self, "_layerdiffuse_base_unet")
                and hasattr(self, "_layerdiffuse_active_timesteps")
                and _timestep_key(timestep) not in self._layerdiffuse_active_timesteps
            ):
                base_unet = self._layerdiffuse_base_unet
                base_unet.to(device=sample.device, dtype=sample.dtype)
                return base_unet(sample, timestep, *args, **kwargs)

            condition = _match_extra_concat_condition(self._layerdiffuse_extra_concat_condition, sample)
            sample = torch.cat([sample, condition], dim=1)
            if hasattr(self, "conv_in") and sample.shape[1] != self.conv_in.in_channels:
                raise ValueError(
                    f"UNet conv_in expects {self.conv_in.in_channels} channels, got {sample.shape[1]}."
                )
            return self._layerdiffuse_original_forward(sample, timestep, *args, **kwargs)

        unet.forward = types.MethodType(forward_with_extra_concat, unet)

    unet._layerdiffuse_extra_concat_condition = condition_latents
    if base_unet is not None:
        unet.__dict__["_layerdiffuse_base_unet"] = base_unet
    if active_timesteps is not None:
        unet._layerdiffuse_active_timesteps = {_timestep_key(timestep) for timestep in active_timesteps}
    return unet


def disable_unet_extra_concat_condition(unet):
    if hasattr(unet, "_layerdiffuse_original_forward"):
        unet.forward = unet._layerdiffuse_original_forward
        delattr(unet, "_layerdiffuse_original_forward")
    if hasattr(unet, "_layerdiffuse_extra_concat_condition"):
        delattr(unet, "_layerdiffuse_extra_concat_condition")
    if hasattr(unet, "_layerdiffuse_base_unet"):
        del unet.__dict__["_layerdiffuse_base_unet"]
    if hasattr(unet, "_layerdiffuse_active_timesteps"):
        delattr(unet, "_layerdiffuse_active_timesteps")
    return unet


def get_kwargs_encoder():
    pass


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def load_lora_to_unet(unet, model_path, frames=1, use_control=False):
    module_mapping_sd15 = {0: 'input_blocks.1.1.transformer_blocks.0.attn1', 1: 'input_blocks.1.1.transformer_blocks.0.attn2', 2: 'input_blocks.2.1.transformer_blocks.0.attn1', 3: 'input_blocks.2.1.transformer_blocks.0.attn2', 4: 'input_blocks.4.1.transformer_blocks.0.attn1', 5: 'input_blocks.4.1.transformer_blocks.0.attn2', 6: 'input_blocks.5.1.transformer_blocks.0.attn1', 7: 'input_blocks.5.1.transformer_blocks.0.attn2', 8: 'input_blocks.7.1.transformer_blocks.0.attn1', 9: 'input_blocks.7.1.transformer_blocks.0.attn2', 10: 'input_blocks.8.1.transformer_blocks.0.attn1', 11: 'input_blocks.8.1.transformer_blocks.0.attn2', 12: 'output_blocks.3.1.transformer_blocks.0.attn1', 13: 'output_blocks.3.1.transformer_blocks.0.attn2', 14: 'output_blocks.4.1.transformer_blocks.0.attn1', 15: 'output_blocks.4.1.transformer_blocks.0.attn2', 16: 'output_blocks.5.1.transformer_blocks.0.attn1', 17: 'output_blocks.5.1.transformer_blocks.0.attn2', 18: 'output_blocks.6.1.transformer_blocks.0.attn1', 19: 'output_blocks.6.1.transformer_blocks.0.attn2', 20: 'output_blocks.7.1.transformer_blocks.0.attn1', 21: 'output_blocks.7.1.transformer_blocks.0.attn2', 22: 'output_blocks.8.1.transformer_blocks.0.attn1', 23: 'output_blocks.8.1.transformer_blocks.0.attn2', 24: 'output_blocks.9.1.transformer_blocks.0.attn1', 25: 'output_blocks.9.1.transformer_blocks.0.attn2', 26: 'output_blocks.10.1.transformer_blocks.0.attn1', 27: 'output_blocks.10.1.transformer_blocks.0.attn2', 28: 'output_blocks.11.1.transformer_blocks.0.attn1', 29: 'output_blocks.11.1.transformer_blocks.0.attn2', 30: 'middle_block.1.transformer_blocks.0.attn1', 31: 'middle_block.1.transformer_blocks.0.attn2'}

    sd15_to_diffusers = {
        'input_blocks.1.1.transformer_blocks.0.attn1': 'down_blocks.0.attentions.0.transformer_blocks.0.attn1', 
        'input_blocks.1.1.transformer_blocks.0.attn2': 'down_blocks.0.attentions.0.transformer_blocks.0.attn2', 
        'input_blocks.2.1.transformer_blocks.0.attn1': 'down_blocks.0.attentions.1.transformer_blocks.0.attn1', 
        'input_blocks.2.1.transformer_blocks.0.attn2': 'down_blocks.0.attentions.1.transformer_blocks.0.attn2', 
        'input_blocks.4.1.transformer_blocks.0.attn1': 'down_blocks.1.attentions.0.transformer_blocks.0.attn1', 
        'input_blocks.4.1.transformer_blocks.0.attn2': 'down_blocks.1.attentions.0.transformer_blocks.0.attn2', 
        'input_blocks.5.1.transformer_blocks.0.attn1': 'down_blocks.1.attentions.1.transformer_blocks.0.attn1', 
        'input_blocks.5.1.transformer_blocks.0.attn2': 'down_blocks.1.attentions.1.transformer_blocks.0.attn2', 
        'input_blocks.7.1.transformer_blocks.0.attn1': 'down_blocks.2.attentions.0.transformer_blocks.0.attn1', 
        'input_blocks.7.1.transformer_blocks.0.attn2': 'down_blocks.2.attentions.0.transformer_blocks.0.attn2', 
        'input_blocks.8.1.transformer_blocks.0.attn1': 'down_blocks.2.attentions.1.transformer_blocks.0.attn1', 
        'input_blocks.8.1.transformer_blocks.0.attn2': 'down_blocks.2.attentions.1.transformer_blocks.0.attn2', 
        'output_blocks.3.1.transformer_blocks.0.attn1': "up_blocks.1.attentions.0.transformer_blocks.0.attn1", 
        'output_blocks.3.1.transformer_blocks.0.attn2': "up_blocks.1.attentions.0.transformer_blocks.0.attn2", 
        'output_blocks.4.1.transformer_blocks.0.attn1': "up_blocks.1.attentions.1.transformer_blocks.0.attn1", 
        'output_blocks.4.1.transformer_blocks.0.attn2': "up_blocks.1.attentions.1.transformer_blocks.0.attn2", 
        'output_blocks.5.1.transformer_blocks.0.attn1': "up_blocks.1.attentions.2.transformer_blocks.0.attn1", 
        'output_blocks.5.1.transformer_blocks.0.attn2': "up_blocks.1.attentions.2.transformer_blocks.0.attn2", 
        'output_blocks.6.1.transformer_blocks.0.attn1': "up_blocks.2.attentions.0.transformer_blocks.0.attn1", 
        'output_blocks.6.1.transformer_blocks.0.attn2': "up_blocks.2.attentions.0.transformer_blocks.0.attn2", 
        'output_blocks.7.1.transformer_blocks.0.attn1': "up_blocks.2.attentions.1.transformer_blocks.0.attn1", 
        'output_blocks.7.1.transformer_blocks.0.attn2': "up_blocks.2.attentions.1.transformer_blocks.0.attn2", 
        'output_blocks.8.1.transformer_blocks.0.attn1': "up_blocks.2.attentions.2.transformer_blocks.0.attn1", 
        'output_blocks.8.1.transformer_blocks.0.attn2': "up_blocks.2.attentions.2.transformer_blocks.0.attn2", 
        'output_blocks.9.1.transformer_blocks.0.attn1': "up_blocks.3.attentions.0.transformer_blocks.0.attn1", 
        'output_blocks.9.1.transformer_blocks.0.attn2': "up_blocks.3.attentions.0.transformer_blocks.0.attn2", 
        'output_blocks.10.1.transformer_blocks.0.attn1': "up_blocks.3.attentions.1.transformer_blocks.0.attn1", 
        'output_blocks.10.1.transformer_blocks.0.attn2': "up_blocks.3.attentions.1.transformer_blocks.0.attn2", 
        'output_blocks.11.1.transformer_blocks.0.attn1': "up_blocks.3.attentions.2.transformer_blocks.0.attn1", 
        'output_blocks.11.1.transformer_blocks.0.attn2': "up_blocks.3.attentions.2.transformer_blocks.0.attn2", 
        'middle_block.1.transformer_blocks.0.attn1': "mid_block.attentions.0.transformer_blocks.0.attn1", 
        'middle_block.1.transformer_blocks.0.attn2': "mid_block.attentions.0.transformer_blocks.0.attn2",
    }

    layer_list = []
    for i in range(32):
        real_key = module_mapping_sd15[i]
        diffuser_key = sd15_to_diffusers[real_key]
        attn_module: Attention = get_attr(unet, diffuser_key)
        if isinstance(attn_module.processor, IPAdapterAttnProcessor2_0):
            u = IPAdapterAttnShareProcessor2_0(attn_module, frames=frames, use_control=use_control).to(unet.dtype)
        elif isinstance(attn_module.processor, IPAdapterAttnProcessor):
            u = IPAdapterAttnShareProcessor(attn_module, frames=frames, use_control=use_control).to(unet.dtype)
        elif isinstance(attn_module.processor, AttnProcessor2_0):
            u = AttentionSharingProcessor2_0(attn_module, frames=frames, use_control=use_control).to(unet.dtype)
        else:
            u = AttentionSharingProcessor(attn_module, frames=frames, use_control=use_control).to(unet.dtype)
        u = u.to(unet.device)
        layer_list.append(u)
        attn_module.set_processor(u)
    
    loader = LoraLoader(layer_list, use_control=use_control)
    lora_state_dict = load_file(model_path)
    loader.load_state_dict(lora_state_dict, strict=False)

    return loader.kwargs_encoder
