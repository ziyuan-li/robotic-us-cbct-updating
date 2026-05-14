#!/usr/bin/env python3
"""Run bidirectional ultrasound deformation inference with USCorUNet.

The script reads an adjacent ultrasound pair, predicts forward/backward dense
flows, and saves the cross-warped frames:

- ``I1_to_I0.png``: ``I1`` warped to the ``I0`` grid;
- ``I0_to_I1.png``: ``I0`` warped to the ``I1`` grid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from infer_utils import (
    autocast_context,
    build_uscorunet_input_5ch,
    ensure_paths_exist,
    load_uscorunet_model,
    log_info,
    make_base_grid,
    read_gray_float01,
    save_gray_float01_png,
    setup_inference_device,
    warp_tensor,
)


@dataclass(frozen=True)
class Config:
    """Runtime configuration for one ultrasound-pair demo."""

    input_dir: Path = Path("examples/ultrasound/case_01")
    checkpoint: Path = Path("models/uscorunet_base.pth")
    output_dir: Path | None = None
    device: str = "auto"
    use_amp: bool = True
    align_corners: bool = True
    input_i0_name: str = "I0.png"
    input_i1_name: str = "I1.png"
    output_i1_to_i0_name: str = "I1_to_I0.png"
    output_i0_to_i1_name: str = "I0_to_I1.png"


@torch.inference_mode()
def run_bidirectional_us_warp(
    *,
    model: torch.nn.Module,
    device: torch.device,
    i0_01,
    i1_01,
    use_amp: bool,
    align_corners: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict bidirectional flow and warp the ultrasound pair."""
    if i0_01.shape != i1_01.shape:
        raise ValueError(f"Input shape mismatch: I0={i0_01.shape}, I1={i1_01.shape}")

    height, width = i0_01.shape
    base_grid = make_base_grid(height, width, device)
    x_np = build_uscorunet_input_5ch(i0_01, i1_01)

    x = torch.from_numpy(x_np).unsqueeze(0).to(device, non_blocking=True)
    i0_t = torch.from_numpy(i0_01).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
    i1_t = torch.from_numpy(i1_01).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)

    with autocast_context(device=device, enabled=use_amp, dtype=torch.float16):
        f01, f10 = model(x)
        i0_prime = warp_tensor(i1_t, f01, base_grid, align_corners=align_corners)
        i1_prime = warp_tensor(i0_t, f10, base_grid, align_corners=align_corners)

    return i0_prime, i1_prime


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Config.input_dir, help="Directory containing I0.png and I1.png.")
    parser.add_argument("--checkpoint", type=Path, default=Config.checkpoint, help="USCorUNet checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to --input-dir.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--align-corners", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--i0-name", default=Config.input_i0_name, help="Reference ultrasound file name.")
    parser.add_argument("--i1-name", default=Config.input_i1_name, help="Deformed ultrasound file name.")
    parser.add_argument("--out-i1-to-i0-name", default=Config.output_i1_to_i0_name, help="Output name for I1 warped to I0.")
    parser.add_argument("--out-i0-to-i1-name", default=Config.output_i0_to_i1_name, help="Output name for I0 warped to I1.")
    args = parser.parse_args()

    return Config(
        input_dir=args.input_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        use_amp=not args.no_amp,
        align_corners=args.align_corners,
        input_i0_name=args.i0_name,
        input_i1_name=args.i1_name,
        output_i1_to_i0_name=args.out_i1_to_i0_name,
        output_i0_to_i1_name=args.out_i0_to_i1_name,
    )


def main() -> None:
    cfg = parse_args()
    output_dir = cfg.output_dir or cfg.input_dir

    input_i0_path = cfg.input_dir / cfg.input_i0_name
    input_i1_path = cfg.input_dir / cfg.input_i1_name
    output_i1_to_i0_path = output_dir / cfg.output_i1_to_i0_name
    output_i0_to_i1_path = output_dir / cfg.output_i0_to_i1_name

    ensure_paths_exist([input_i0_path, input_i1_path, cfg.checkpoint])
    i0_01, _ = read_gray_float01(input_i0_path)
    i1_01, _ = read_gray_float01(input_i1_path)

    device = setup_inference_device(cfg.device)
    model = load_uscorunet_model(cfg.checkpoint, device, in_ch=5, strict=False)

    i0_prime, i1_prime = run_bidirectional_us_warp(
        model=model,
        device=device,
        i0_01=i0_01,
        i1_01=i1_01,
        use_amp=cfg.use_amp,
        align_corners=cfg.align_corners,
    )

    save_gray_float01_png(output_i1_to_i0_path, i0_prime)
    save_gray_float01_png(output_i0_to_i1_path, i1_prime)
    log_info(f"Saved: {output_i1_to_i0_path}")
    log_info(f"Saved: {output_i0_to_i1_path}")


if __name__ == "__main__":
    main()
