#!/usr/bin/env python3
"""Download USCorUNet checkpoints from Hugging Face.

By default, checkpoints are downloaded from ``ziyuan-li/uscorunet`` into the
local ``models/`` directory. If the model repository is private, set ``HF_TOKEN``
or ``HUGGINGFACE_HUB_TOKEN`` in the environment before running this script.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_REPO_ID = "ziyuan-li/uscorunet"
CHECKPOINTS = {
    "uscorunet_base.pth": "uscorunet_base.pth",
    "uscorunet_probe_adapted.pth": "uscorunet_probe_adapted.pth",
    "uscorunet_external_adapted.pth": "uscorunet_external_adapted.pth",
}


def _token_from_env() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _download_file(url: str, output_path: Path, token: str | None) -> None:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urlopen(request) as response, output_path.open("wb") as handle:
            total = int(response.headers.get("Content-Length", "0") or "0")
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100.0 / total
                    print(f"\r  {output_path.name}: {pct:5.1f}%", end="", flush=True)
            print()
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError(
                "Hugging Face denied access. If the repository is private, set "
                "HF_TOKEN or HUGGINGFACE_HUB_TOKEN with read permission."
            ) from exc
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face model repository ID.")
    parser.add_argument("--revision", default="main", help="Repository revision, branch, or tag.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Local checkpoint directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = _token_from_env()

    for remote_name, local_name in CHECKPOINTS.items():
        url = f"https://huggingface.co/{args.repo_id}/resolve/{args.revision}/{remote_name}"
        output_path = args.output_dir / local_name
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[SKIP] {output_path} already exists.")
            continue
        print(f"[DOWNLOAD] {remote_name} -> {output_path}")
        _download_file(url, output_path, token)

    print("[DONE] Checkpoints are ready.")


if __name__ == "__main__":
    main()
