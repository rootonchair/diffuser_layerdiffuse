import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.convert_xl_fg2ble import convert_file


DEFAULT_INPUT = Path("/mnt/disks/workspace/sd-forge-layerdiffuse/layer_xl_fgble2bg.safetensors")
DEFAULT_OUTPUT = Path("weights/diffuser_layer_xl_fgble2bg.safetensors")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Forge layer_xl_fgble2bg.safetensors to Diffusers keys.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to Forge layer_xl_fgble2bg.safetensors.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Converted Diffusers delta output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, key_count = convert_file(args.input, args.output, expected_input_channels=12)
    print(f"Saved {key_count} converted tensors to {output_path}")


if __name__ == "__main__":
    main()
