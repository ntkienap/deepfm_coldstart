import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_DATA = os.environ.get("PATH_ROOT_DATA", str(BASE_DIR))


def process_single_image(args):
    """Worker function executed in separate processes for maximum multi-core CPU throughput."""
    img_path, out_crop_dir, out_resize_dir, size_image, force = args
    stem = Path(img_path).stem

    crop_0 = out_crop_dir / f"{stem}_0.jpg"
    crop_1 = out_crop_dir / f"{stem}_1.jpg"
    crop_2 = out_crop_dir / f"{stem}_2.jpg"
    crop_3 = out_crop_dir / f"{stem}_3.jpg"
    crop_4 = out_crop_dir / f"{stem}_4.jpg"
    resize_img = out_resize_dir / f"{stem}.jpg"

    # Resume capability: skip if all 5 crops exist and are non-empty
    if not force:
        if (crop_0.exists() and crop_1.exists() and crop_2.exists() and 
            crop_3.exists() and crop_4.exists() and crop_0.stat().st_size > 500):
            return stem, "skipped"

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            min_size = min(w, h)
            size_result = size_image + 0.33 * size_image
            w_result = round(w / min_size * size_result)
            h_result = round(h / min_size * size_result)

            # Resize keeping aspect ratio
            resized = img.resize((w_result, h_result), Image.Resampling.BILINEAR)
            resized.save(resize_img, "JPEG", quality=90)

            # 4 Corners: 1=top-left, 2=top-right, 3=bottom-right, 4=bottom-left
            resized.crop((0, 0, size_image, size_image)).save(crop_1, "JPEG", quality=90)
            resized.crop((w_result - size_image, 0, w_result, size_image)).save(crop_2, "JPEG", quality=90)
            resized.crop((w_result - size_image, h_result - size_image, w_result, h_result)).save(crop_3, "JPEG", quality=90)
            resized.crop((0, h_result - size_image, size_image, h_result)).save(crop_4, "JPEG", quality=90)

            # Center crop: 0=center
            left = round((w_result - size_image) / 2)
            top = round((h_result - size_image) / 2)
            resized.crop((left, top, left + size_image, top + size_image)).save(crop_0, "JPEG", quality=90)

        return stem, "cropped"
    except Exception:
        return stem, "error"


def main():
    parser = argparse.ArgumentParser(description="High-Speed Multi-Core Image Resizing & 5-Corner Cropping")
    parser.add_argument("--root-data", default=DEFAULT_ROOT_DATA, help="Root data folder")
    parser.add_argument("--size", type=int, default=256, help="Target crop size (default: 256)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8), help="CPU worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to crop (for testing)")
    parser.add_argument("--force", action="store_true", help="Re-crop even if output already exists")
    args = parser.parse_args()

    root_data = Path(args.root_data).resolve()
    path_image_origin = root_data / "image_orgin" / "movielens"
    path_image_resize = root_data / "image_resize" / "movielens"
    path_image_crop = root_data / "image_crop" / "movielens"

    path_image_resize.mkdir(parents=True, exist_ok=True)
    path_image_crop.mkdir(parents=True, exist_ok=True)

    if not path_image_origin.exists():
        print(f"Error: Origin folder not found at {path_image_origin}", file=sys.stderr)
        sys.exit(1)

    all_images = sorted(list(path_image_origin.glob("*.jpg")))
    total_found = len(all_images)
    print(f"Found {total_found} poster images in {path_image_origin}")

    if args.limit:
        all_images = all_images[:args.limit]
        print(f"Limited processing to {len(all_images)} images.")

    if not all_images:
        print("No images to process.")
        return

    # Prepare multiprocessing arguments
    task_args = [
        (str(p), path_image_crop, path_image_resize, args.size, args.force)
        for p in all_images
    ]

    print(f"Starting cropping with {args.workers} CPU worker processes...")
    start_time = time.time()
    cropped_count = 0
    skipped_count = 0
    error_count = 0

    pbar = tqdm(total=len(all_images), desc="Cropping 5-pieces", unit="movie") if tqdm else None

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for stem, status in executor.map(process_single_image, task_args, chunksize=50):
            if status == "cropped":
                cropped_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                error_count += 1

            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    "cropped": cropped_count,
                    "skipped": skipped_count,
                    "error": error_count
                })

    if pbar:
        pbar.close()

    elapsed = time.time() - start_time
    print("\nFinished Cropping Process!")
    print(f"- Total movies: {len(all_images)}")
    print(f"- Successfully cropped: {cropped_count} ({cropped_count * 5} cropped images generated)")
    print(f"- Already existing (skipped): {skipped_count}")
    if error_count > 0:
        print(f"- Errors: {error_count}")
    print(f"- Total elapsed time: {elapsed:.2f}s ({len(all_images) / max(elapsed, 0.001):.1f} movies/s)")
    print(f"- Output cropped folder: {path_image_crop}")
    print(f"- Output resized folder: {path_image_resize}")


if __name__ == "__main__":
    main()
