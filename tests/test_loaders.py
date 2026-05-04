import pytest
import torch

from layer_diffuse.loaders import (
    disable_unet_extra_concat_condition,
    enable_unet_extra_concat_condition,
    merge_delta_weights_into_unet,
    merge_sdxl_concat_delta_weights_into_unet,
)


class DummyPipe:
    def __init__(self, unet):
        self.unet = unet


class DummyUnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(4, 2, kernel_size=1)


def test_merge_sdxl_concat_delta_expands_conv_in_and_preserves_base_channels():
    pipe = DummyPipe(DummyUnet())
    original_weight = pipe.unet.conv_in.weight.detach().clone()
    delta = {
        "conv_in.weight": torch.ones(2, 8, 1, 1),
        "conv_in.bias": torch.ones(2),
    }

    merge_sdxl_concat_delta_weights_into_unet(pipe, delta)

    assert pipe.unet.conv_in.in_channels == 8
    assert torch.allclose(pipe.unet.conv_in.weight[:, :4], original_weight + 1)
    assert torch.allclose(pipe.unet.conv_in.weight[:, 4:], torch.ones(2, 4, 1, 1))


def test_merge_sdxl_concat_delta_accepts_already_expanded_conv_in():
    pipe = DummyPipe(DummyUnet())
    delta = {
        "conv_in.weight": torch.zeros(2, 8, 1, 1),
        "conv_in.bias": torch.zeros(2),
    }

    merge_sdxl_concat_delta_weights_into_unet(pipe, delta)
    merge_sdxl_concat_delta_weights_into_unet(pipe, delta)

    assert pipe.unet.conv_in.in_channels == 8


def test_merge_sdxl_concat_delta_supports_fg_blend_to_background_channels():
    pipe = DummyPipe(DummyUnet())
    delta = {
        "conv_in.weight": torch.zeros(2, 12, 1, 1),
        "conv_in.bias": torch.zeros(2),
    }

    merge_sdxl_concat_delta_weights_into_unet(pipe, delta, extra_channels=8)

    assert pipe.unet.conv_in.in_channels == 12


def test_merge_delta_weights_into_unet_still_rejects_unknown_keys():
    pipe = DummyPipe(DummyUnet())
    with pytest.raises(AssertionError):
        merge_delta_weights_into_unet(pipe, {"missing.weight": torch.zeros(1)})


class ForwardUnet(torch.nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        self.last_sample = None

    def forward(self, sample, timestep, return_dict=False):
        self.last_sample = sample
        return (sample,)


def test_extra_concat_condition_repeats_for_classifier_free_guidance_batches():
    unet = ForwardUnet()
    condition = torch.ones(1, 4, 2, 2)
    sample = torch.zeros(2, 4, 2, 2)

    enable_unet_extra_concat_condition(unet, condition)
    output = unet(sample, torch.tensor(1), return_dict=False)[0]

    assert output.shape == (2, 8, 2, 2)
    assert torch.allclose(unet.last_sample[:, :4], sample)
    assert torch.allclose(unet.last_sample[:, 4:], torch.ones(2, 4, 2, 2))

    disable_unet_extra_concat_condition(unet)
    assert not hasattr(unet, "_layerdiffuse_original_forward")
    assert unet(sample, torch.tensor(1), return_dict=False)[0].shape == (2, 4, 2, 2)


def test_extra_concat_condition_rejects_ambiguous_batch_multiples():
    unet = ForwardUnet()
    condition = torch.ones(2, 4, 2, 2)
    sample = torch.zeros(6, 4, 2, 2)

    enable_unet_extra_concat_condition(unet, condition)

    with pytest.raises(ValueError, match="half the sample batch"):
        unet(sample, torch.tensor(1), return_dict=False)


def test_extra_concat_condition_supports_foreground_and_blend_conditions():
    unet = ForwardUnet(in_channels=12)
    foreground = torch.ones(1, 4, 2, 2)
    blend = torch.full((1, 4, 2, 2), 2.0)
    sample = torch.zeros(2, 4, 2, 2)

    enable_unet_extra_concat_condition(unet, torch.cat([foreground, blend], dim=1))
    output = unet(sample, torch.tensor(1), return_dict=False)[0]

    assert output.shape == (2, 12, 2, 2)
    assert torch.allclose(unet.last_sample[:, :4], sample)
    assert torch.allclose(unet.last_sample[:, 4:8], torch.ones(2, 4, 2, 2))
    assert torch.allclose(unet.last_sample[:, 8:], torch.full((2, 4, 2, 2), 2.0))


def test_extra_concat_condition_routes_inactive_timesteps_to_base_unet():
    unet = ForwardUnet(in_channels=12)
    base_unet = ForwardUnet(in_channels=4)
    condition = torch.ones(1, 8, 2, 2)
    sample = torch.zeros(1, 4, 2, 2)

    enable_unet_extra_concat_condition(unet, condition, base_unet=base_unet, active_timesteps=[10])
    active_output = unet(sample, torch.tensor(10), return_dict=False)[0]
    inactive_output = unet(sample, torch.tensor(5), return_dict=False)[0]

    assert active_output.shape == (1, 12, 2, 2)
    assert inactive_output.shape == (1, 4, 2, 2)
    assert torch.allclose(unet.last_sample[:, 4:], torch.ones(1, 8, 2, 2))
    assert torch.allclose(base_unet.last_sample, sample)
