import torch.nn as nn
import torch.nn.functional as F
import torch
import einops
from typing import Optional, List

from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.models.attention_processor import Attention


class HookerLayers(torch.nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.layers = torch.nn.ModuleList(layer_list)


class AdditionalAttentionCondsEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks_0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 64*64*256

        self.blocks_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 32*32*256

        self.blocks_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 16*16*256

        self.blocks_3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
        )  # 8*8*256

        self.blks = [self.blocks_0, self.blocks_1, self.blocks_2, self.blocks_3]

    def __call__(self, h):
        results = {}
        for b in self.blks:
            h = b(h)
            results[int(h.shape[2]) * int(h.shape[3])] = h
        return results


class LoraLoader(torch.nn.Module):
    def __init__(self, layer_list, use_control=False):
        super().__init__()
        self.hookers = HookerLayers(layer_list)

        if use_control:
            self.kwargs_encoder = AdditionalAttentionCondsEncoder()
        else:
            self.kwargs_encoder = None


class LoRALinearLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 256):
        super().__init__()
        self.down = torch.nn.Linear(in_features, rank, bias=False)
        self.up = torch.nn.Linear(rank, out_features, bias=False)

    def forward(self, h, org):
        org_weight = org.weight.to(h)
        if hasattr(org, 'bias'):
            org_bias = org.bias.to(h) if org.bias is not None else None
        else:
            org_bias = None
        down_weight = self.down.weight
        up_weight = self.up.weight
        final_weight = org_weight + torch.mm(up_weight, down_weight)
        return torch.nn.functional.linear(h, final_weight, org_bias)


class AttentionSharingProcessor(nn.Module):
    def __init__(self, module, frames=2, rank=256, use_control=False):
        super().__init__()

        self.heads = module.heads
        self.frames = frames
        self.original_module = [module]
        q_in_channels, q_out_channels = module.to_q.in_features, module.to_q.out_features
        k_in_channels, k_out_channels = module.to_k.in_features, module.to_k.out_features
        v_in_channels, v_out_channels = module.to_v.in_features, module.to_v.out_features
        o_in_channels, o_out_channels = module.to_out[0].in_features, module.to_out[0].out_features

        hidden_size = k_out_channels

        self.to_q_lora = [LoRALinearLayer(q_in_channels, q_out_channels, rank) for _ in range(self.frames)]
        self.to_k_lora = [LoRALinearLayer(k_in_channels, k_out_channels, rank) for _ in range(self.frames)]
        self.to_v_lora = [LoRALinearLayer(v_in_channels, v_out_channels, rank) for _ in range(self.frames)]
        self.to_out_lora = [LoRALinearLayer(o_in_channels, o_out_channels, rank) for _ in range(self.frames)]

        self.to_q_lora = torch.nn.ModuleList(self.to_q_lora)
        self.to_k_lora = torch.nn.ModuleList(self.to_k_lora)
        self.to_v_lora = torch.nn.ModuleList(self.to_v_lora)
        self.to_out_lora = torch.nn.ModuleList(self.to_out_lora)

        self.temporal_i = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_n = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.temporal_q = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_k = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_v = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_o = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.control_convs = None

        if use_control:
            self.control_convs = [torch.nn.Sequential(
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(256, hidden_size, kernel_size=1),
            ) for _ in range(self.frames)]
            self.control_convs = torch.nn.ModuleList(self.control_convs)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.Tensor] = None,
        layerdiffuse_control_signals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # LayerDiffuse main logic
        modified_hidden_states = einops.rearrange(hidden_states, '(b f) d c -> f b d c', f=self.frames)

        if self.control_convs is not None:
            context_dim = int(modified_hidden_states.shape[2])
            control_outs = []
            for f in range(self.frames):
                control_signal = layerdiffuse_control_signals[context_dim].to(modified_hidden_states)
                control = self.control_convs[f](control_signal)
                control = einops.rearrange(control, 'b c h w -> b (h w) c')
                control_outs.append(control)
            control_outs = torch.stack(control_outs, dim=0)
            modified_hidden_states = modified_hidden_states + control_outs.to(modified_hidden_states)

        if encoder_hidden_states is None:
            framed_context = modified_hidden_states
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            framed_context = einops.rearrange(encoder_hidden_states, '(b f) d c -> f b d c', f=self.frames)


        attn_outs = []
        for f in range(self.frames):
            fcf = framed_context[f]

            query = self.to_q_lora[f](modified_hidden_states[f], attn.to_q)
            key = self.to_k_lora[f](fcf, attn.to_k)
            value = self.to_v_lora[f](fcf, attn.to_v)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            output = torch.bmm(attention_probs, value)
            output = attn.batch_to_head_dim(output)
            output = self.to_out_lora[f](output, attn.to_out[0])
            output = attn.to_out[1](output)
            attn_outs.append(output)

        attn_outs = torch.stack(attn_outs, dim=0)
        modified_hidden_states = modified_hidden_states + attn_outs.to(modified_hidden_states)
        modified_hidden_states = einops.rearrange(modified_hidden_states, 'f b d c -> (b f) d c', f=self.frames)
        modified_hidden_states = modified_hidden_states / attn.rescale_output_factor

        x = modified_hidden_states
        x = self.temporal_n(x)
        x = self.temporal_i(x)
        d = x.shape[1]

        x = einops.rearrange(x, "(b f) d c -> (b d) f c", f=self.frames)

        query = self.temporal_q(x)
        key = self.temporal_k(x)
        value = self.temporal_v(x)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        x = torch.bmm(attention_probs, value)
        x = attn.batch_to_head_dim(x)

        x = self.temporal_o(x)
        x = einops.rearrange(x, "(b d) f c -> (b f) d c", d=d)

        modified_hidden_states = modified_hidden_states + x

        hidden_states = modified_hidden_states - hidden_states

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states


class IPAdapterAttnShareProcessor(nn.Module):
    def __init__(self, module, frames=2, rank=256):
        super().__init__()

        self.heads = module.heads
        self.frames = frames
        self.original_module = [module]
        q_in_channels, q_out_channels = module.to_q.in_features, module.to_q.out_features
        k_in_channels, k_out_channels = module.to_k.in_features, module.to_k.out_features
        v_in_channels, v_out_channels = module.to_v.in_features, module.to_v.out_features
        o_in_channels, o_out_channels = module.to_out[0].in_features, module.to_out[0].out_features

        hidden_size = k_out_channels

        self.to_q_lora = [LoRALinearLayer(q_in_channels, q_out_channels, rank) for _ in range(self.frames)]
        self.to_k_lora = [LoRALinearLayer(k_in_channels, k_out_channels, rank) for _ in range(self.frames)]
        self.to_v_lora = [LoRALinearLayer(v_in_channels, v_out_channels, rank) for _ in range(self.frames)]
        self.to_out_lora = [LoRALinearLayer(o_in_channels, o_out_channels, rank) for _ in range(self.frames)]

        self.to_q_lora = torch.nn.ModuleList(self.to_q_lora)
        self.to_k_lora = torch.nn.ModuleList(self.to_k_lora)
        self.to_v_lora = torch.nn.ModuleList(self.to_v_lora)
        self.to_out_lora = torch.nn.ModuleList(self.to_out_lora)

        self.temporal_i = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_n = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.temporal_q = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_k = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_v = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_o = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)

        # IP-Adapter part
        self.scale = module.processor.scale
        self.num_tokens = module.processor.num_tokens

        self.to_k_ip = module.processor.to_k_ip
        self.to_v_ip = module.processor.to_v_ip
    
    def _fuse_ip_adapter(
        self,
        attn: Attention,
        batch_size: int,
        query: torch.Tensor,
        hidden_states: torch.Tensor,
        ip_hidden_states: Optional[torch.Tensor] = None,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ):
        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = attn.head_to_batch_dim(ip_key)
                        ip_value = attn.head_to_batch_dim(ip_value)

                        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
                        _current_ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
                        _current_ip_hidden_states = attn.batch_to_head_dim(_current_ip_hidden_states)

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)

                        hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)

                    ip_key = attn.head_to_batch_dim(ip_key)
                    ip_value = attn.head_to_batch_dim(ip_value)

                    ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
                    current_ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
                    current_ip_hidden_states = attn.batch_to_head_dim(current_ip_hidden_states)

                    hidden_states = hidden_states + scale * current_ip_hidden_states

        return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                print(deprecation_message)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)


        modified_hidden_states = einops.rearrange(hidden_states, '(b f) d c -> f b d c', f=self.frames)

        if encoder_hidden_states is None:
            framed_context = modified_hidden_states
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            framed_context = einops.rearrange(encoder_hidden_states, '(b f) d c -> f b d c', f=self.frames)


        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)


        attn_outs = []
        for f in range(self.frames):
            fcf = framed_context[f]

            query = self.to_q_lora[f](modified_hidden_states[f], attn.to_q)
            key = self.to_k_lora[f](fcf, attn.to_k)
            value = self.to_v_lora[f](fcf, attn.to_v)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            output = torch.bmm(attention_probs, value)
            output = attn.batch_to_head_dim(output)

            # IP-Adapter process
            output = self._fuse_ip_adapter(
                attn=attn,
                batch_size=batch_size,
                query=query,
                hidden_states=output,
                ip_hidden_states=ip_hidden_states,
                ip_adapter_masks=ip_adapter_masks
            )

            output = self.to_out_lora[f](output, attn.to_out[0])
            output = attn.to_out[1](output)
            attn_outs.append(output)

        attn_outs = torch.stack(attn_outs, dim=0)
        modified_hidden_states = modified_hidden_states + attn_outs.to(modified_hidden_states)
        modified_hidden_states = einops.rearrange(modified_hidden_states, 'f b d c -> (b f) d c', f=self.frames)

        x = modified_hidden_states
        x = self.temporal_n(x)
        x = self.temporal_i(x)
        d = x.shape[1]

        x = einops.rearrange(x, "(b f) d c -> (b d) f c", f=self.frames)

        query = self.temporal_q(x)
        key = self.temporal_k(x)
        value = self.temporal_v(x)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        x = torch.bmm(attention_probs, value)
        x = attn.batch_to_head_dim(x)

        x = self.temporal_o(x)
        x = einops.rearrange(x, "(b d) f c -> (b f) d c", d=d)

        modified_hidden_states = modified_hidden_states + x

        hidden_states = modified_hidden_states - hidden_states

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states


class AttentionSharingProcessor2_0(nn.Module):
    def __init__(self, module, frames=2, rank=256, use_control=False):
        super().__init__()

        self.heads = module.heads
        self.frames = frames
        self.original_module = [module]
        q_in_channels, q_out_channels = module.to_q.in_features, module.to_q.out_features
        k_in_channels, k_out_channels = module.to_k.in_features, module.to_k.out_features
        v_in_channels, v_out_channels = module.to_v.in_features, module.to_v.out_features
        o_in_channels, o_out_channels = module.to_out[0].in_features, module.to_out[0].out_features

        hidden_size = k_out_channels

        self.to_q_lora = [LoRALinearLayer(q_in_channels, q_out_channels, rank) for _ in range(self.frames)]
        self.to_k_lora = [LoRALinearLayer(k_in_channels, k_out_channels, rank) for _ in range(self.frames)]
        self.to_v_lora = [LoRALinearLayer(v_in_channels, v_out_channels, rank) for _ in range(self.frames)]
        self.to_out_lora = [LoRALinearLayer(o_in_channels, o_out_channels, rank) for _ in range(self.frames)]

        self.to_q_lora = torch.nn.ModuleList(self.to_q_lora)
        self.to_k_lora = torch.nn.ModuleList(self.to_k_lora)
        self.to_v_lora = torch.nn.ModuleList(self.to_v_lora)
        self.to_out_lora = torch.nn.ModuleList(self.to_out_lora)

        self.temporal_i = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_n = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.temporal_q = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_k = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_v = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_o = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.control_convs = None

        if use_control:
            self.control_convs = [torch.nn.Sequential(
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(256, hidden_size, kernel_size=1),
            ) for _ in range(self.frames)]
            self.control_convs = torch.nn.ModuleList(self.control_convs)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.Tensor] = None,
        layerdiffuse_control_signals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # LayerDiffuse main logic
        modified_hidden_states = einops.rearrange(hidden_states, '(b f) d c -> f b d c', f=self.frames)

        if self.control_convs is not None:
            context_dim = int(modified_hidden_states.shape[2])
            control_outs = []
            for f in range(self.frames):
                control_signal = layerdiffuse_control_signals[context_dim].to(modified_hidden_states)
                control = self.control_convs[f](control_signal)
                control = einops.rearrange(control, 'b c h w -> b (h w) c')
                control_outs.append(control)
            control_outs = torch.stack(control_outs, dim=0)
            modified_hidden_states = modified_hidden_states + control_outs.to(modified_hidden_states)

        if encoder_hidden_states is None:
            framed_context = modified_hidden_states
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            framed_context = einops.rearrange(encoder_hidden_states, '(b f) d c -> f b d c', f=self.frames)


        attn_outs = []
        for f in range(self.frames):
            fcf = framed_context[f]
            frame_batch_size = fcf.size(0)

            query = self.to_q_lora[f](modified_hidden_states[f], attn.to_q)
            key = self.to_k_lora[f](fcf, attn.to_k)
            value = self.to_v_lora[f](fcf, attn.to_v)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(frame_batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(frame_batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(frame_batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            output = output.transpose(1, 2).reshape(frame_batch_size, -1, attn.heads * head_dim)
            output = output.to(query.dtype)

            output = self.to_out_lora[f](output, attn.to_out[0])
            output = attn.to_out[1](output)
            attn_outs.append(output)

        attn_outs = torch.stack(attn_outs, dim=0)
        modified_hidden_states = modified_hidden_states + attn_outs.to(modified_hidden_states)
        modified_hidden_states = einops.rearrange(modified_hidden_states, 'f b d c -> (b f) d c', f=self.frames)
        modified_hidden_states = modified_hidden_states / attn.rescale_output_factor

        x = modified_hidden_states
        x = self.temporal_n(x)
        x = self.temporal_i(x)
        d = x.shape[1]

        x = einops.rearrange(x, "(b f) d c -> (b d) f c", f=self.frames)

        temporal_batch_size = x.size(0)

        query = self.temporal_q(x)
        key = self.temporal_k(x)
        value = self.temporal_v(x)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(temporal_batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(temporal_batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(temporal_batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        x = x.transpose(1, 2).reshape(temporal_batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        x = self.temporal_o(x)
        x = einops.rearrange(x, "(b d) f c -> (b f) d c", d=d)

        modified_hidden_states = modified_hidden_states + x

        hidden_states = modified_hidden_states - hidden_states

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states


class IPAdapterAttnShareProcessor2_0(nn.Module):
    def __init__(self, module, frames=2, rank=256):
        super().__init__()

        self.heads = module.heads
        self.frames = frames
        self.original_module = [module]
        q_in_channels, q_out_channels = module.to_q.in_features, module.to_q.out_features
        k_in_channels, k_out_channels = module.to_k.in_features, module.to_k.out_features
        v_in_channels, v_out_channels = module.to_v.in_features, module.to_v.out_features
        o_in_channels, o_out_channels = module.to_out[0].in_features, module.to_out[0].out_features

        hidden_size = k_out_channels

        self.to_q_lora = [LoRALinearLayer(q_in_channels, q_out_channels, rank) for _ in range(self.frames)]
        self.to_k_lora = [LoRALinearLayer(k_in_channels, k_out_channels, rank) for _ in range(self.frames)]
        self.to_v_lora = [LoRALinearLayer(v_in_channels, v_out_channels, rank) for _ in range(self.frames)]
        self.to_out_lora = [LoRALinearLayer(o_in_channels, o_out_channels, rank) for _ in range(self.frames)]

        self.to_q_lora = torch.nn.ModuleList(self.to_q_lora)
        self.to_k_lora = torch.nn.ModuleList(self.to_k_lora)
        self.to_v_lora = torch.nn.ModuleList(self.to_v_lora)
        self.to_out_lora = torch.nn.ModuleList(self.to_out_lora)

        self.temporal_i = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_n = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.temporal_q = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_k = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_v = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.temporal_o = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)

        # IP-Adapter part
        self.scale = module.processor.scale
        self.num_tokens = module.processor.num_tokens

        self.to_k_ip = module.processor.to_k_ip
        self.to_v_ip = module.processor.to_v_ip
    
    def _fuse_ip_adapter(
        self,
        attn: Attention,
        batch_size: int,
        head_dim: int,
        query: torch.Tensor,
        hidden_states: torch.Tensor,
        ip_hidden_states: Optional[torch.Tensor] = None,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ):
        # for ip-adapter

        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                        _current_ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                        _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
                            batch_size, -1, attn.heads * head_dim
                        )
                        _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                        hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)

                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    current_ip_hidden_states = F.scaled_dot_product_attention(
                        query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                    )

                    current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                        batch_size, -1, attn.heads * head_dim
                    )
                    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                    hidden_states = hidden_states + scale * current_ip_hidden_states
        
        return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                print(deprecation_message)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)


        modified_hidden_states = einops.rearrange(hidden_states, '(b f) d c -> f b d c', f=self.frames)

        if encoder_hidden_states is None:
            framed_context = modified_hidden_states
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            framed_context = einops.rearrange(encoder_hidden_states, '(b f) d c -> f b d c', f=self.frames)


        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)


        attn_outs = []
        for f in range(self.frames):
            fcf = framed_context[f]
            frame_batch_size = fcf.size(0)

            query = self.to_q_lora[f](modified_hidden_states[f], attn.to_q)
            key = self.to_k_lora[f](fcf, attn.to_k)
            value = self.to_v_lora[f](fcf, attn.to_v)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(frame_batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(frame_batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(frame_batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            output = output.transpose(1, 2).reshape(frame_batch_size, -1, attn.heads * head_dim)
            output = output.to(query.dtype)

            # IP-Adapter process
            output = self._fuse_ip_adapter(
                attn=attn,
                batch_size=frame_batch_size,
                head_dim=head_dim,
                query=query,
                hidden_states=output,
                ip_hidden_states=ip_hidden_states,
                ip_adapter_masks=ip_adapter_masks
            )

            output = self.to_out_lora[f](output, attn.to_out[0])
            output = attn.to_out[1](output)
            attn_outs.append(output)

        attn_outs = torch.stack(attn_outs, dim=0)
        modified_hidden_states = modified_hidden_states + attn_outs.to(modified_hidden_states)
        modified_hidden_states = einops.rearrange(modified_hidden_states, 'f b d c -> (b f) d c', f=self.frames)

        x = modified_hidden_states
        x = self.temporal_n(x)
        x = self.temporal_i(x)
        d = x.shape[1]

        x = einops.rearrange(x, "(b f) d c -> (b d) f c", f=self.frames)

        temporal_batch_size = x.size(0)

        query = self.temporal_q(x)
        key = self.temporal_k(x)
        value = self.temporal_v(x)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(temporal_batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(temporal_batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(temporal_batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        x = x.transpose(1, 2).reshape(temporal_batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        x = self.temporal_o(x)
        x = einops.rearrange(x, "(b d) f c -> (b f) d c", d=d)

        modified_hidden_states = modified_hidden_states + x

        hidden_states = modified_hidden_states - hidden_states

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states