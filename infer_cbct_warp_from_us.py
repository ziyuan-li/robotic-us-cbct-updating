#!/usr/bin/env python3
"""Warp a CBCT slice using dense deformation estimated from ultrasound.

The script reads an ultrasound pair and a source CBCT slice at t0. USCorUNet
predicts bidirectional ultrasound deformation, and the backward flow ``F10`` is
used to synthesize the CBCT slice at t1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_utils import (
    autocast_context,
    build_uscorunet_input_5ch,
    ensure_paths_exist,
    load_uscorunet_model,
    log_info,
    log_warn,
    make_base_grid,
    pad_images_to_common_hw,
    read_gray_float01,
    save_float01_like_dtype_png,
    setup_inference_device,
    warp_tensor,
)


@dataclass(frozen=True)
class Config:
    """Runtime configuration for one CBCT-warping demo."""

    case_dir: Path = Path("examples/us_cbct/case_02")
    checkpoint: Path = Path("models/uscorunet_probe_adapted.pth")
    output_dir: Path | None = None
    device: str = "auto"
    use_amp: bool = True
    align_corners: bool = True
    resize_ct_to_us: bool = True
    us_t0_name: str = "US_I0.png"
    us_t1_name: str = "US_I1.png"
    ct_t0_name: str = "CBCT_I0.png"
    out_ct_t1_pred_name: str = "CBCT_I1_pred.png"


@torch.inference_mode()
def run_us_guided_cbct_warp(
    *,
    model: torch.nn.Module,
    device: torch.device,
    us0_01: np.ndarray,
    us1_01: np.ndarray,
    ct0_01: np.ndarray,
    use_amp: bool,
    align_corners: bool,
) -> np.ndarray:
    """Predict ultrasound deformation and warp ``CBCT_t0`` to ``CBCT_t1``."""
    if not (us0_01.shape == us1_01.shape == ct0_01.shape):
        raise ValueError(
            "All inputs must share the same shape before inference. "
            f"Got US_t0={us0_01.shape}, US_t1={us1_01.shape}, CBCT_t0={ct0_01.shape}"
        )

    height, width = us0_01.shape
    base_grid = make_base_grid(height, width, device)
    x_np = build_uscorunet_input_5ch(us0_01, us1_01)

    x = torch.from_numpy(x_np).unsqueeze(0).to(device, non_blocking=True)
    ct0_t = torch.from_numpy(ct0_01).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)

    with autocast_context(device=device, enabled=use_amp, dtype=torch.float16):
        _, f10 = model(x)
        ct1_pred = warp_tensor(ct0_t, f10, base_grid, align_corners=align_corners)

    out = ct1_pred.squeeze().clamp(0.0, 1.0).to(torch.float32).cpu().numpy()
    return np.ascontiguousarray(out, dtype=np.float32)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=Config.case_dir, help="Directory containing US and CBCT inputs.")
    parser.add_argument("--checkpoint", type=Path, default=Config.checkpoint, help="USCorUNet checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to --case-dir.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--align-corners", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-resize-ct-to-us", action="store_true", help="Do not resize CBCT_t0 to the US image size.")
    parser.add_argument("--us-t0-name", default=Config.us_t0_name, help="Reference ultrasound file name.")
    parser.add_argument("--us-t1-name", default=Config.us_t1_name, help="Deformed ultrasound file name.")
    parser.add_argument("--ct-t0-name", default=Config.ct_t0_name, help="Source CBCT file name.")
    parser.add_argument("--out-name", default=Config.out_ct_t1_pred_name, help="Output CBCT prediction file name.")
    args = parser.parse_args()

    return Config(
        case_dir=args.case_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        use_amp=not args.no_amp,
        align_corners=args.align_corners,
        resize_ct_to_us=not args.no_resize_ct_to_us,
        us_t0_name=args.us_t0_name,
        us_t1_name=args.us_t1_name,
        ct_t0_name=args.ct_t0_name,
        out_ct_t1_pred_name=args.out_name,
    )


def main() -> None:
    cfg = parse_args()
    output_dir = cfg.output_dir or cfg.case_dir

    us_t0_path = cfg.case_dir / cfg.us_t0_name
    us_t1_path = cfg.case_dir / cfg.us_t1_name
    ct_t0_path = cfg.case_dir / cfg.ct_t0_name
    out_ct_t1_pred_path = output_dir / cfg.out_ct_t1_pred_name

    ensure_paths_exist([us_t0_path, us_t1_path, ct_t0_path, cfg.checkpoint])
    us0_01, _ = read_gray_float01(us_t0_path)
    us1_01, _ = read_gray_float01(us_t1_path)
    ct0_01, ct_dtype = read_gray_float01(ct_t0_path)

    if cfg.resize_ct_to_us and ct0_01.shape != us0_01.shape:
        target_h, target_w = us0_01.shape
        ct0_01 = cv2.resize(ct0_01, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        ct0_01 = np.ascontiguousarray(ct0_01, dtype=np.float32)
        log_warn(f"Resized CBCT_t0 to match ultrasound shape: {(target_h, target_w)}")

    (us0_01, us1_01, ct0_01), _, _ = pad_images_to_common_hw(
        [us0_01, us1_01, ct0_01],
        pad_value=0.0,
    )

    device = setup_inference_device(cfg.device)
    model = load_uscorunet_model(cfg.checkpoint, device, in_ch=5, strict=False)

    ct1_pred_01 = run_us_guided_cbct_warp(
        model=model,
        device=device,
        us0_01=us0_01,
        us1_01=us1_01,
        ct0_01=ct0_01,
        use_amp=cfg.use_amp,
        align_corners=cfg.align_corners,
    )

    save_float01_like_dtype_png(out_ct_t1_pred_path, ct1_pred_01, ct_dtype)
    log_info(f"Saved: {out_ct_t1_pred_path}")


if __name__ == "__main__":
    main()
