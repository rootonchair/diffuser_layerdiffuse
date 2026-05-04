import torch

from scripts.convert_xl_fg2ble import convert_ldm_unet_delta_key, convert_state_dict


def test_convert_ldm_unet_delta_key_examples():
    assert (
        convert_ldm_unet_delta_key("diffusion_model.input_blocks.0.0.weight::diff::0")
        == "conv_in.weight"
    )
    assert (
        convert_ldm_unet_delta_key("diffusion_model.input_blocks.3.0.op.bias::diff::0")
        == "down_blocks.0.downsamplers.0.conv.bias"
    )
    assert (
        convert_ldm_unet_delta_key("diffusion_model.middle_block.2.out_layers.3.weight::diff::0")
        == "mid_block.resnets.1.conv2.weight"
    )
    assert (
        convert_ldm_unet_delta_key("diffusion_model.output_blocks.2.2.conv.weight::diff::0")
        == "up_blocks.0.upsamplers.0.conv.weight"
    )


def test_convert_state_dict_remaps_and_validates():
    state_dict = {
        "diffusion_model.input_blocks.0.0.weight::diff::0": torch.zeros(320, 8, 3, 3),
        "diffusion_model.input_blocks.0.0.bias::diff::0": torch.zeros(320),
        "diffusion_model.input_blocks.1.0.in_layers.0.weight::diff::0": torch.zeros(320),
        "diffusion_model.input_blocks.1.0.emb_layers.1.bias::diff::0": torch.zeros(320),
        "diffusion_model.output_blocks.0.0.skip_connection.weight::diff::0": torch.zeros(1280, 2560, 1, 1),
        "diffusion_model.out.2.bias::diff::0": torch.zeros(4),
    }

    converted = convert_state_dict(state_dict)

    assert converted["conv_in.weight"].shape == (320, 8, 3, 3)
    assert "down_blocks.0.resnets.0.norm1.weight" in converted
    assert "down_blocks.0.resnets.0.time_emb_proj.bias" in converted
    assert "up_blocks.0.resnets.0.conv_shortcut.weight" in converted
    assert "conv_out.bias" in converted
    assert all("::diff::" not in key for key in converted)


def test_convert_state_dict_accepts_fgble2bg_12_channel_input():
    state_dict = {
        "diffusion_model.input_blocks.0.0.weight::diff::0": torch.zeros(320, 12, 3, 3),
        "diffusion_model.input_blocks.0.0.bias::diff::0": torch.zeros(320),
    }

    converted = convert_state_dict(state_dict, expected_input_channels=12)

    assert converted["conv_in.weight"].shape == (320, 12, 3, 3)
