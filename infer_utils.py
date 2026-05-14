"""Shared utilities for USCorUNet inference.

The helpers in this module keep the demo scripts small and explicit:

- image I/O and grayscale conversion;
- normalization to float32 ``[0, 1]``;
- construction of the 5-channel USCorUNet input;
- dense flow warping with ``torch.grid_sample``;
- checkpoint loading with common state-dict formats.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from uscorunet import USCorUNet


def log_info(message: str) -> None:
    """Print an info-level message."""
    print(f"[INFO] {message}")


def log_warn(message: str) -> None:
    """Print a warning-level message."""
    print(f"[WARN] {message}")


def read_image_any(path: Path) -> np.ndarray:
    """Read an image with OpenCV while preserving bit depth."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return np.ascontiguousarray(image)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV image array to a single grayscale channel."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0]
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def _integer_dtype_max(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def image_to_float01(gray: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    """Normalize a 2D grayscale image to contiguous float32 values in ``[0, 1]``."""
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape={gray.shape}")

    original_dtype = gray.dtype
    if gray.dtype == np.uint8:
        image01 = gray.astype(np.float32) / 255.0
    elif gray.dtype == np.uint16:
        image01 = gray.astype(np.float32) / 65535.0
    elif np.issubdtype(gray.dtype, np.integer):
        denom = max(_integer_dtype_max(gray.dtype), 1.0)
        image01 = gray.astype(np.float32) / denom
    else:
        image01 = gray.astype(np.float32)

    image01 = np.clip(image01, 0.0, 1.0)
    return np.ascontiguousarray(image01, dtype=np.float32), original_dtype


def read_gray_float01(path: Path) -> tuple[np.ndarray, np.dtype]:
    """Read an image, convert it to grayscale, and normalize it to ``[0, 1]``."""
    raw = read_image_any(path)
    gray = to_grayscale(raw)
    return image_to_float01(gray)


def save_gray_float01_png(path: Path, tensor_or_array: torch.Tensor | np.ndarray) -> None:
    """Save a grayscale ``[0, 1]`` image as an 8-bit PNG."""
    if isinstance(tensor_or_array, torch.Tensor):
        image01 = tensor_or_array.detach().squeeze().clamp(0.0, 1.0).to(torch.float32).cpu().numpy()
    else:
        image01 = np.asarray(tensor_or_array, dtype=np.float32).squeeze()
        image01 = np.clip(image01, 0.0, 1.0)

    if image01.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image after squeeze, got {image01.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = (image01 * 255.0 + 0.5).astype(np.uint8)
    if not cv2.imwrite(str(path), image_u8):
        raise OSError(f"Failed to write PNG: {path}")


def save_float01_like_dtype_png(path: Path, image01: np.ndarray, reference_dtype: np.dtype) -> None:
    """Save a float ``[0, 1]`` image as uint8 or uint16, matching a reference dtype."""
    if image01.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got {image01.shape}")

    image01 = np.clip(np.asarray(image01, dtype=np.float32), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)

    if reference_dtype == np.uint16:
        out = (image01 * 65535.0 + 0.5).astype(np.uint16)
    else:
        out = (image01 * 255.0 + 0.5).astype(np.uint8)

    if not cv2.imwrite(str(path), out):
        raise OSError(f"Failed to write PNG: {path}")


def pad_to_hw(arr: np.ndarray, height: int, width: int, value: float | int = 0) -> np.ndarray:
    """Pad a 2D/3D array on the bottom and right to ``(height, width)``."""
    if arr.ndim == 2:
        h, w = arr.shape
        out = np.full((height, width), value, dtype=arr.dtype)
        out[:h, :w] = arr
        return out
    if arr.ndim == 3:
        h, w, c = arr.shape
        out = np.full((height, width, c), value, dtype=arr.dtype)
        out[:h, :w, :] = arr
        return out
    raise ValueError(f"pad_to_hw only supports 2D/3D arrays, got shape={arr.shape}")


def pad_images_to_common_hw(
    images: Sequence[np.ndarray],
    pad_value: float | int = 0,
) -> tuple[list[np.ndarray], int, int]:
    """Pad 2D images to their common maximum height and width."""
    if not images:
        raise ValueError("pad_images_to_common_hw received an empty image list.")
    for index, image in enumerate(images):
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image at index {index}, got shape={image.shape}")

    target_h = max(image.shape[0] for image in images)
    target_w = max(image.shape[1] for image in images)
    padded = [pad_to_hw(image, target_h, target_w, value=pad_value) for image in images]
    return padded, target_h, target_w


def np_grad_mag(image01: np.ndarray) -> np.ndarray:
    """Compute a simple forward-difference gradient magnitude."""
    if image01.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape={image01.shape}")

    image01 = np.ascontiguousarray(image01, dtype=np.float32)
    gy = np.zeros_like(image01, dtype=np.float32)
    gx = np.zeros_like(image01, dtype=np.float32)
    gy[1:, :] = image01[1:, :] - image01[:-1, :]
    gx[:, 1:] = image01[:, 1:] - image01[:, :-1]
    return np.sqrt(np.maximum(gx * gx + gy * gy, 0.0)).astype(np.float32)


def build_uscorunet_input_5ch(i0_01: np.ndarray, i1_01: np.ndarray) -> np.ndarray:
    """Build ``[I0, I1, I1 - I0, grad(I0), grad(I1)]`` for USCorUNet."""
    if i0_01.shape != i1_01.shape:
        raise ValueError(f"Input shape mismatch: I0={i0_01.shape}, I1={i1_01.shape}")

    i0_01 = np.ascontiguousarray(i0_01, dtype=np.float32)
    i1_01 = np.ascontiguousarray(i1_01, dtype=np.float32)
    diff = (i1_01 - i0_01).astype(np.float32)
    g0 = np_grad_mag(i0_01)
    g1 = np_grad_mag(i1_01)
    stacked = np.stack([i0_01, i1_01, diff, g0, g1], axis=0)
    return np.ascontiguousarray(stacked, dtype=np.float32)




def make_base_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Create a normalized ``grid_sample`` base grid with shape ``(1, H, W, 2)``."""
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)


def warp_tensor(
    src: torch.Tensor,
    flow_px: torch.Tensor,
    base_grid: torch.Tensor,
    *,
    align_corners: bool = True,
    padding_mode: str = "zeros",
    mode: str = "bilinear",
) -> torch.Tensor:
    """Warp ``src`` with a dense pixel-unit flow field in ``(dx, dy)`` order."""
    if src.ndim != 4:
        raise ValueError(f"src must be (B, C, H, W), got {tuple(src.shape)}")
    if flow_px.ndim != 4 or flow_px.shape[1] != 2:
        raise ValueError(f"flow_px must be (B, 2, H, W), got {tuple(flow_px.shape)}")

    _, _, height, width = flow_px.shape
    fx = flow_px[:, 0] * (2.0 / max(width - 1, 1))
    fy = flow_px[:, 1] * (2.0 / max(height - 1, 1))
    grid = base_grid + torch.stack([fx, fy], dim=-1)

    return F.grid_sample(
        src,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def setup_inference_device(device_name: str = "auto") -> torch.device:
    """Select the inference device."""
    requested = device_name.lower().strip()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{device_name}'. Use one of: auto, cpu, cuda.")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    device = torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        log_info("Using CUDA for inference.")
    else:
        log_warn("Using CPU for inference. This can be slow for the demo checkpoints.")
    return device


def autocast_context(
    *,
    device: torch.device,
    enabled: bool,
    dtype: torch.dtype = torch.float16,
):
    """Return a CUDA autocast context when enabled, otherwise a no-op context."""
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def _safe_torch_load_weights(path: str | Path):
    """Load a checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_uscorunet_model(
    ckpt_path: str | Path,
    device: torch.device,
    *,
    in_ch: int = 5,
    strict: bool = False,
) -> torch.nn.Module:
    """Load a USCorUNet checkpoint in raw, ``model``, or ``state_dict`` format."""
    model = USCorUNet(in_ch=in_ch).to(device).eval()
    ckpt = _safe_torch_load_weights(ckpt_path)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")

    if any(key.startswith("module.") for key in state.keys()):
        state = {key[7:]: value for key, value in state.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state, strict=strict)
    if missing_keys:
        preview = ", ".join(missing_keys[:10])
        suffix = " ..." if len(missing_keys) > 10 else ""
        log_warn(f"Missing checkpoint keys ({len(missing_keys)}): {preview}{suffix}")
    if unexpected_keys:
        preview = ", ".join(unexpected_keys[:10])
        suffix = " ..." if len(unexpected_keys) > 10 else ""
        log_warn(f"Unexpected checkpoint keys ({len(unexpected_keys)}): {preview}{suffix}")
    return model




def ensure_paths_exist(paths: Iterable[Path]) -> None:
    """Raise ``FileNotFoundError`` if any path is missing."""
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
