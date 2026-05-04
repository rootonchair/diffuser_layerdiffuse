import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.mark.gpu
def test_xl_fg2ble_example_gpu_smoke(tmp_path):
    if os.getenv("LAYERDIFFUSE_RUN_GPU_SMOKE") != "1":
        pytest.skip("Set LAYERDIFFUSE_RUN_GPU_SMOKE=1 to run model-download GPU smoke tests.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if not Path("weights/diffuser_layer_xl_fg2ble.safetensors").exists():
        pytest.skip("Converted fg2ble weight is not available.")

    output = tmp_path / "result_xl_fg2ble.png"
    subprocess.run(
        [
            sys.executable,
            "test_diffusers_xl_fg2ble.py",
            "--steps",
            "1",
            "--width",
            "512",
            "--height",
            "512",
            "--output",
            str(output),
        ],
        check=True,
    )
    assert output.exists()
    image = Image.open(output)
    assert image.getextrema() != ((0, 0), (0, 0), (0, 0))


@pytest.mark.gpu
def test_xl_fgble2bg_example_gpu_smoke(tmp_path):
    if os.getenv("LAYERDIFFUSE_RUN_GPU_SMOKE") != "1":
        pytest.skip("Set LAYERDIFFUSE_RUN_GPU_SMOKE=1 to run model-download GPU smoke tests.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    required = [
        Path("weights/diffuser_layer_xl_fgble2bg.safetensors"),
        Path("assets/sdxl_fg_cond_detailed.png"),
        Path("assets/sdxl_fg2ble_detailed_default_scheduler.png"),
    ]
    for path in required:
        if not path.exists():
            pytest.skip(f"Required fgble2bg smoke input is not available: {path}")

    output = tmp_path / "result_xl_fgble2bg.png"
    subprocess.run(
        [
            sys.executable,
            "test_diffusers_xl_fgble2bg.py",
            "--steps",
            "1",
            "--width",
            "512",
            "--height",
            "512",
            "--output",
            str(output),
        ],
        check=True,
    )
    assert output.exists()
    image = Image.open(output)
    assert image.getextrema() != ((0, 0), (0, 0), (0, 0))
