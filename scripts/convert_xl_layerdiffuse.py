import argparse
import re
from pathlib import Path

from safetensors.torch import load_file, save_file


MODE_CONFIGS = {
    "fg2ble": {
        "expected_input_channels": 8,
        "output": Path("weights/diffuser_layer_xl_fg2ble.safetensors"),
    },
    "bg2ble": {
        "expected_input_channels": 8,
        "output": Path("weights/diffuser_layer_xl_bg2ble.safetensors"),
    },
    "fgble2bg": {
        "expected_input_channels": 12,
        "output": Path("weights/diffuser_layer_xl_fgble2bg.safetensors"),
    },
    "bgble2fg": {
        "expected_input_channels": 12,
        "output": Path("weights/diffuser_layer_xl_bgble2fg.safetensors"),
    },
}
_DIFF_SUFFIX_RE = re.compile(r"::diff::\d+$")


def normalize_forge_delta_key(key):
    key = _DIFF_SUFFIX_RE.sub("", key)
    if key.startswith("diffusion_model."):
        key = key[len("diffusion_model.") :]
    if key.startswith("model.diffusion_model."):
        key = key[len("model.diffusion_model.") :]
    return key


def _convert_resnet_path(path):
    return (
        path.replace("in_layers.0", "norm1")
        .replace("in_layers.2", "conv1")
        .replace("out_layers.0", "norm2")
        .replace("out_layers.3", "conv2")
        .replace("emb_layers.1", "time_emb_proj")
        .replace("skip_connection", "conv_shortcut")
    )


def convert_ldm_unet_delta_key(key, layers_per_block=2):
    key = normalize_forge_delta_key(key)

    if key.startswith("input_blocks.0.0."):
        return key.replace("input_blocks.0.0", "conv_in", 1)
    if key.startswith("out.0."):
        return key.replace("out.0", "conv_norm_out", 1)
    if key.startswith("out.2."):
        return key.replace("out.2", "conv_out", 1)

    match = re.match(r"input_blocks\.(\d+)\.0\.op\.(weight|bias)$", key)
    if match:
        block_index = int(match.group(1))
        block_id = (block_index - 1) // (layers_per_block + 1)
        return f"down_blocks.{block_id}.downsamplers.0.conv.{match.group(2)}"

    match = re.match(r"input_blocks\.(\d+)\.0\.(.+)$", key)
    if match:
        block_index = int(match.group(1))
        block_id = (block_index - 1) // (layers_per_block + 1)
        layer_in_block_id = (block_index - 1) % (layers_per_block + 1)
        return (
            f"down_blocks.{block_id}.resnets.{layer_in_block_id}."
            f"{_convert_resnet_path(match.group(2))}"
        )

    match = re.match(r"input_blocks\.(\d+)\.1\.(.+)$", key)
    if match:
        block_index = int(match.group(1))
        block_id = (block_index - 1) // (layers_per_block + 1)
        layer_in_block_id = (block_index - 1) % (layers_per_block + 1)
        return f"down_blocks.{block_id}.attentions.{layer_in_block_id}.{match.group(2)}"

    match = re.match(r"middle_block\.(0|2)\.(.+)$", key)
    if match:
        resnet_id = 0 if match.group(1) == "0" else 1
        return f"mid_block.resnets.{resnet_id}.{_convert_resnet_path(match.group(2))}"

    match = re.match(r"middle_block\.1\.(.+)$", key)
    if match:
        return f"mid_block.attentions.0.{match.group(1)}"

    match = re.match(r"output_blocks\.(\d+)\.0\.(.+)$", key)
    if match:
        block_index = int(match.group(1))
        block_id = block_index // (layers_per_block + 1)
        layer_in_block_id = block_index % (layers_per_block + 1)
        return (
            f"up_blocks.{block_id}.resnets.{layer_in_block_id}."
            f"{_convert_resnet_path(match.group(2))}"
        )

    match = re.match(r"output_blocks\.(\d+)\.1\.(.+)$", key)
    if match:
        block_index = int(match.group(1))
        block_id = block_index // (layers_per_block + 1)
        layer_in_block_id = block_index % (layers_per_block + 1)
        return f"up_blocks.{block_id}.attentions.{layer_in_block_id}.{match.group(2)}"

    match = re.match(r"output_blocks\.(\d+)\.\d+\.conv\.(weight|bias)$", key)
    if match:
        block_index = int(match.group(1))
        block_id = block_index // (layers_per_block + 1)
        return f"up_blocks.{block_id}.upsamplers.0.conv.{match.group(2)}"

    raise ValueError(f"Unsupported Forge SDXL delta key: {key}")


def convert_state_dict(state_dict, expected_input_channels=8):
    converted = {}
    for key, value in state_dict.items():
        new_key = convert_ldm_unet_delta_key(key)
        if new_key in converted:
            raise ValueError(f"Multiple source keys convert to {new_key}.")
        converted[new_key] = value
    validate_converted_state_dict(converted, expected_input_channels=expected_input_channels)
    return converted


def validate_converted_state_dict(state_dict, expected_input_channels=8):
    if "conv_in.weight" not in state_dict:
        raise ValueError("Converted state dict is missing conv_in.weight.")
    if tuple(state_dict["conv_in.weight"].shape[1:]) != (expected_input_channels, 3, 3):
        raise ValueError(
            f"Expected conv_in.weight to have shape (out_channels, {expected_input_channels}, 3, 3), "
            f"got {tuple(state_dict['conv_in.weight'].shape)}."
        )
    for key in state_dict:
        if key.startswith("diffusion_model.") or "::diff::" in key:
            raise ValueError(f"Unconverted Forge key remained in output: {key}")
    return True


def convert_file(input_path, output_path, expected_input_channels=8):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Forge weight not found: {input_path}. Pass --input pointing to a Forge weight.")

    state_dict = load_file(str(input_path), device="cpu")
    converted = convert_state_dict(state_dict, expected_input_channels=expected_input_channels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path))
    return output_path, len(converted)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Forge SDXL LayerDiffuse weights to Diffusers keys.")
    parser.add_argument("--mode", choices=sorted(MODE_CONFIGS), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Path to the Forge SDXL LayerDiffuse weight.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Converted Diffusers delta output path. Defaults to the selected mode's weights/ filename.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = MODE_CONFIGS[args.mode]
    output = args.output if args.output is not None else config["output"]
    output_path, key_count = convert_file(
        args.input,
        output,
        expected_input_channels=config["expected_input_channels"],
    )
    print(f"Saved {key_count} converted tensors to {output_path}")


if __name__ == "__main__":
    main()
