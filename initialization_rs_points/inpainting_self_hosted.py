#!/usr/bin/env python3
"""
Self-hosted FLUX Fill Inpainting Script

This script mirrors the CLI of `inpainting.py` but targets a self-hosted
Flux Fill service that returns base64-encoded images synchronously.
"""

import argparse
import base64
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

import requests
from PIL import Image

from .inpainting import build_inpainting_jobs


class SelfHostedFluxFillAPI:
    """Client for a self-hosted Flux Fill server."""

    DEFAULT_URL = "YOU_PATH"
    DEFAULT_MAX_SEQUENCE_LENGTH = 512

    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("FLUX_FILL_URL") or self.DEFAULT_URL
        self.token = token or os.getenv("FLUX_FILL_TOKEN")
        if not self.token:
            raise ValueError(
                "Authorization token required. Provide via --api-key or FLUX_FILL_TOKEN env variable."
            )

    @staticmethod
    def _infer_mimetype(path: Path) -> str:
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"

    @staticmethod
    def _read_image_shape(path: Path) -> Dict[str, int]:
        with Image.open(path) as img:
            width, height = img.size
        return {"width": width, "height": height}

    @staticmethod
    def _pil_format(path: Path) -> str:
        ext = path.suffix.lower()
        return {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".bmp": "BMP",
            ".tif": "TIFF",
            ".tiff": "TIFF",
            ".webp": "WEBP",
        }.get(ext, "PNG")

    def inpaint(
        self,
        prompt: str,
        image_path: str,
        output_path: str,
        mask_path: Optional[str] = None,
        steps: int = 50,
        guidance: float = 30.0,
        output_format: str = "jpeg",
        safety_tolerance: int = 2,
        timeout: int = 300,
        seed: int = 0,
    ):
        """Submit inpainting request and write decoded image to disk."""
        # Convert to paths for consistent handling
        src_image = Path(image_path)
        dst_image = Path(output_path)
        src_mask = Path(mask_path) if mask_path else None

        original_dims = self._read_image_shape(src_image)
        target_size = (512, 512)

        image_format = self._pil_format(src_image)
        with Image.open(src_image) as pil_image:
            resized_image = pil_image.resize(target_size, Image.BILINEAR)
            image_buffer = BytesIO()
            resized_image.save(image_buffer, format=image_format)
            image_buffer.seek(0)

        mask_buffer = None
        if src_mask:
            with Image.open(src_mask) as pil_mask:
                resized_mask = pil_mask.resize(target_size, Image.NEAREST)
                mask_buffer = BytesIO()
                resized_mask.save(mask_buffer, format="PNG")
                mask_buffer.seek(0)

        files = [
            (
                "image",
                (
                    src_image.name,
                    image_buffer,
                    self._infer_mimetype(src_image),
                ),
            )
        ]
        if mask_buffer is not None:
            files.append(
                (
                    "mask",
                    (
                        src_mask.name,
                        mask_buffer,
                        self._infer_mimetype(src_mask),
                    ),
                )
            )

        data = {
            "prompt": prompt or "",
            "guidance_scale": f"{guidance}",
            "num_inference_steps": f"{steps}",
            "max_sequence_length": f"{self.DEFAULT_MAX_SEQUENCE_LENGTH}",
            "seed": f"{seed}",
            "width": f"{target_size[0]}",
            "height": f"{target_size[1]}",
        }

        headers = {"Authorization": f"Bearer {self.token}"}

        response = requests.post(
            self.base_url,
            headers=headers,
            data=data,
            files=files,
            timeout=timeout,
        )
        # print(f"Response: {response.json()}")
        response.raise_for_status()
        payload = response.json()
        image_base64 = payload.get("image_base64")
        if not image_base64:
            raise ValueError(f"image_base64 missing in response: {payload}")

        dst_image.parent.mkdir(parents=True, exist_ok=True)
        decoded = base64.b64decode(image_base64)
        with Image.open(BytesIO(decoded)) as result_img:
            restored = result_img.resize(
                (original_dims["width"], original_dims["height"]),
                Image.BILINEAR,
            )
            restored.save(dst_image, format=result_img.format or image_format)


def main():
    parser = argparse.ArgumentParser(description="Self-hosted Flux Fill Inpainting Script")
    parser.add_argument(
        "--image",
        "--image-path",
        dest="image_path",
        type=str,
        default="/YOU_PATH/render_img/scene_000/real_img",
        help="Path to input image or directory",
    )
    parser.add_argument(
        "--mask",
        "--mask-path",
        dest="mask_path",
        type=str,
        default="/YOU_PATH/render_img/scene_000/real_mask",
        help="Path to mask image or directory (optional if image has alpha channel)",
    )
    parser.add_argument("--prompt", type=str, help="Text prompt describing desired changes")
    parser.add_argument(
        "--output",
        type=str,
        default="/YOU_PATH/render_img/scene_000/inpainted_img",
        help="Path to save output image or directory",
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=30, help="Guidance scale (default: 30)")
    parser.add_argument(
        "--format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="Retained for CLI parity; output format is inferred from service response.",
    )
    parser.add_argument(
        "--safety-tolerance",
        type=int,
        default=2,
        help="Unused placeholder for CLI compatibility with inpainting.py.",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Maximum wait time in seconds (default: 300)")
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Authorization token for self-hosted Flux Fill (fallback to FLUX_FILL_TOKEN env variable).",
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)
    mask_path = Path(args.mask_path) if args.mask_path else None
    output_path = Path(args.output)

    if mask_path and not mask_path.exists():
        print(f"Error: Mask path not found: {mask_path}")
        return 1

    try:
        api = SelfHostedFluxFillAPI(token=args.api_key or None)
        jobs = list(build_inpainting_jobs(image_path, mask_path, output_path, args.format))
        total = len(jobs)

        for idx, (img_file, mask_file, out_path) in enumerate(jobs, start=1):
            print(f"\nProcessing {idx}/{total}: {img_file} -> {out_path}")
            api.inpaint(
                prompt=args.prompt or "",
                image_path=str(img_file),
                output_path=str(out_path),
                mask_path=str(mask_file) if mask_file else None,
                steps=args.steps,
                guidance=args.guidance,
                output_format=args.format,
                safety_tolerance=args.safety_tolerance,
                timeout=args.timeout,
            )

        print("\n✓ Inpainting completed successfully!")
        return 0
    except Exception as exc:
        print(f"\n✗ Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
