#!/usr/bin/env python3
"""
FLUX.1 Fill Inpainting Script

This script uses the FLUX.1 Fill API to perform image inpainting.
It supports two modes:
1. Separate mask: Provide image and mask files separately
2. Alpha channel: Provide a PNG/WebP with transparency

Usage:
    python flux_fill_inpainting.py --image input.jpg --mask mask.jpg --prompt "your prompt" --output result.jpg
    or
    python flux_fill_inpainting.py --image input_with_alpha.png --prompt "your prompt" --output result.jpg
"""

import argparse
import base64
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import requests


class FluxFillAPI:
    """Client for FLUX.1 Fill API"""
    
    BASE_URL = "YOU_HEEP/fill"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API client
        
        Args:
            api_key: BFL API key. If not provided, will read from BFL_API_KEY env variable
        """
        self.api_key = api_key or os.getenv("BFL_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set BFL_API_KEY environment variable or pass it as argument.")
    
    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """
        Encode an image file to base64 string
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def create_inpainting_request(
        self,
        prompt: str,
        image_path: str,
        mask_path: Optional[str] = None,
        steps: int = 50,
        guidance: float = 30,
        output_format: str = "jpeg",
        safety_tolerance: int = 2,
    ) -> str:
        """
        Create an inpainting request
        
        Args:
            prompt: Text prompt describing desired changes
            image_path: Path to input image
            mask_path: Path to mask image (optional if image has alpha channel)
            steps: Number of inference steps (default: 50)
            guidance: Guidance scale (default: 30)
            output_format: Output format - "jpeg" or "png" (default: "jpeg")
            safety_tolerance: Safety tolerance level (default: 2)
            
        Returns:
            Polling URL to check for results
        """
        print("Encoding image to base64...")
        image_base64 = self.encode_image_to_base64(image_path)
        
        payload = {
            "prompt": prompt,
            "image": image_base64,
            "steps": steps,
            "guidance": guidance,
            "output_format": output_format,
            "safety_tolerance": safety_tolerance,
        }
        
        if mask_path:
            print("Encoding mask to base64...")
            mask_base64 = self.encode_image_to_base64(mask_path)
            payload["mask"] = mask_base64
        
        headers = {
            "x-key": self.api_key,
            "Content-Type": "application/json",
        }
        
        print("Sending request to FLUX.1 Fill API...")
        response = requests.post(self.BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        polling_url = result.get("id")
        
        if not polling_url:
            raise ValueError(f"Unexpected response format: {result}")
        
        # Construct full polling URL
        full_polling_url = f"YOU_PATH/get_result?id={polling_url}"
        print(f"Request created. Polling URL: {full_polling_url}")
        
        return full_polling_url
    
    def poll_for_result(self, polling_url: str, timeout: int = 300) -> str:
        """
        Poll for the generation result
        
        Args:
            polling_url: URL to poll for results
            timeout: Maximum time to wait in seconds (default: 300)
            
        Returns:
            URL to download the generated image
        """
        headers = {
            "accept": "application/json",
            "x-key": self.api_key,
        }
        
        start_time = time.time()
        print("Polling for results...")
        
        # Initial wait to allow API to register the request
        time.sleep(1.0)
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Request timed out after {timeout} seconds")
            
            try:
                response = requests.get(polling_url, headers=headers)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # Request not ready yet, wait and retry
                    print("Request not found yet, waiting...")
                    time.sleep(2.0)
                    continue
                else:
                    raise
            
            result = response.json()
            status = result.get("status")
            
            print(f"Status: {status}")
            
            if status == "Ready":
                sample_url = result.get("result", {}).get("sample")
                if not sample_url:
                    raise ValueError(f"No sample URL in result: {result}")
                print("Generation complete!")
                return sample_url
            elif status in ["Error", "Failed"]:
                raise RuntimeError(f"Generation failed: {result}")
            elif status == "Pending":
                time.sleep(0.5)  # Wait before next poll
                continue
            else:
                print(f"Unknown status: {status}, continuing to poll...")
                time.sleep(0.5)
    
    def download_image(self, url: str, output_path: str):
        """
        Download the generated image
        
        Args:
            url: URL to download from
            output_path: Path to save the image
        """
        print(f"Downloading image to {output_path}...")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        print(f"Image saved successfully to {output_path}")
    
    def inpaint(
        self,
        prompt: str,
        image_path: str,
        output_path: str,
        mask_path: Optional[str] = None,
        steps: int = 50,
        guidance: float = 30,
        output_format: str = "jpeg",
        safety_tolerance: int = 2,
        timeout: int = 300,
    ):
        """
        Complete inpainting workflow
        
        Args:
            prompt: Text prompt describing desired changes
            image_path: Path to input image
            output_path: Path to save output image
            mask_path: Path to mask image (optional if image has alpha channel)
            steps: Number of inference steps (default: 50)
            guidance: Guidance scale (default: 30)
            output_format: Output format - "jpeg" or "png" (default: "jpeg")
            safety_tolerance: Safety tolerance level (default: 2)
            timeout: Maximum time to wait in seconds (default: 300)
        """
        # Create request
        polling_url = self.create_inpainting_request(
            prompt=prompt,
            image_path=image_path,
            mask_path=mask_path,
            steps=steps,
            guidance=guidance,
            output_format=output_format,
            safety_tolerance=safety_tolerance,
        )
        
        # Poll for result
        result_url = self.poll_for_result(polling_url, timeout=timeout)
        
        # Download result
        self.download_image(result_url, output_path)


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def build_inpainting_jobs(
    image_path: Path,
    mask_path: Optional[Path],
    output_path: Path,
    output_format: str,
) -> Iterable[Tuple[Path, Optional[Path], Path]]:
    """
    Yield (image, mask, output) tuples for inpainting.
    Handles both single files and directories for images and masks.
    """
    if not image_path.exists():
        raise ValueError(f"Image path not found: {image_path}")

    # Single image file flow
    if image_path.is_file():
        resolved_mask = None
        if mask_path:
            if mask_path.is_dir():
                candidates = [
                    mask_path / image_path.name,
                    *(mask_path / f"{image_path.stem}{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS),
                ]
                for candidate in candidates:
                    if candidate.exists():
                        resolved_mask = candidate
                        break
            elif mask_path.is_file():
                resolved_mask = mask_path
            else:
                raise ValueError(f"Mask path not found: {mask_path}")

        resolved_output = output_path
        if output_path.is_dir():
            resolved_output = output_path / f"{image_path.stem}.{output_format}"
        elif not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        yield image_path, resolved_mask, resolved_output
        return

    # Directory flow
    if not image_path.is_dir():
        raise ValueError(f"Image path must be file or directory: {image_path}")

    if output_path.exists() and output_path.is_file():
        raise ValueError("When processing a directory, --output must be a directory path")
    output_path.mkdir(parents=True, exist_ok=True)

    mask_dir: Optional[Path] = None
    if mask_path:
        if mask_path.is_file():
            raise ValueError("Mask path must be a directory when processing multiple images")
        if not mask_path.exists():
            raise ValueError(f"Mask path not found: {mask_path}")
        mask_dir = mask_path

    images = sorted(p for p in image_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images found in {image_path}")

    for img_file in images:
        resolved_mask: Optional[Path] = None
        if mask_dir:
            # Prefer masks named like mask_000.png, fall back to image-based naming
            parts = img_file.stem.split("_")
            mask_idx = parts[-1] if parts else img_file.stem
            named_candidates = [
                mask_dir / f"mask_{mask_idx}{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS
            ]
            same_name_candidates = [
                mask_dir / img_file.name,
                *(mask_dir / f"{img_file.stem}{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS),
            ]
            for candidate in (*named_candidates, *same_name_candidates):
                if candidate.exists():
                    resolved_mask = candidate
                    break
        elif mask_path:
            # Mask path provided but not directory (handled earlier)
            resolved_mask = mask_path

        resolved_output = output_path / f"{img_file.stem}.{output_format}"
        yield img_file, resolved_mask, resolved_output


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.1 Fill Inpainting Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

        """
    )
    
    parser.add_argument("--image", "--image-path", dest="image_path", type=str, default='/YOU_PATH/render_img/scene_000/real_img', help="Path to input image or directory")
    parser.add_argument("--mask", "--mask-path", dest="mask_path", type=str, default='/YOU_PATH/render_img/scene_000/real_mask', help="Path to mask image or directory (optional if image has alpha channel)")
    parser.add_argument("--prompt", type=str, help="Text prompt describing desired changes")
    parser.add_argument("--output", type=str, default='/YOU_PATH/render_img/scene_000/inpainted_img', help="Path to save output image or directory")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=30, help="Guidance scale (default: 30)")
    parser.add_argument("--format", choices=["jpeg", "png"], default="jpeg", help="Output format (default: jpeg)")
    parser.add_argument("--safety-tolerance", type=int, default=2, help="Safety tolerance level (default: 2)")
    parser.add_argument("--timeout", type=int, default=300, help="Maximum wait time in seconds (default: 300)")
    parser.add_argument("--api-key", type=str, default=None, help="BFL API key (can also use BFL_API_KEY env variable)")
    
    args = parser.parse_args()
    
    # Validate input files
    image_path = Path(args.image_path)
    mask_path = Path(args.mask_path) if args.mask_path else None
    output_path = Path(args.output)

    if mask_path and not mask_path.exists():
        print(f"Error: Mask path not found: {mask_path}")
        return 1
    
    try:
        # Initialize API client
        api = FluxFillAPI(api_key=args.api_key)
        jobs = list(build_inpainting_jobs(image_path, mask_path, output_path, args.format))

        total = len(jobs)
        for idx, (img_file, mask_file, out_path) in enumerate(jobs, start=1):
            print(f"\nProcessing {idx}/{total}: {img_file} -> {out_path}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            api.inpaint(
                prompt=args.prompt,
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
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
