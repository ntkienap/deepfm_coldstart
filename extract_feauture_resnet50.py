import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = Path(__file__).resolve().parent
PATH_ROOT_DATA = os.environ.get("PATH_ROOT_DATA", str(BASE_DIR))

PATH_DATA_CSV = f"{PATH_ROOT_DATA}/movielens/ml-25m/movies.csv"
FILE_DATASET = f"{PATH_ROOT_DATA}/movielens/dataset_image_movielens.csv"

PATH_IMAGE_ORIGIN = f"{PATH_ROOT_DATA}/image_orgin/movielens"
PATH_IMAGE_CROP = f"{PATH_ROOT_DATA}/image_crop/movielens"
PATH_IMAGE_RESIZE = f"{PATH_ROOT_DATA}/image_resize/movielens"

PATH_FEATURE_MAX = f"{PATH_ROOT_DATA}/feature/movielens/ResNet50_max"
PATH_FEATURE_AVG = f"{PATH_ROOT_DATA}/feature/movielens/ResNet50_avg"
PATH_FEATURE_MAX_FINISH = f"{PATH_ROOT_DATA}/feature/movielens/ResNet50_max.finish"
PATH_FEATURE_AVG_FINISH = f"{PATH_ROOT_DATA}/feature/movielens/ResNet50_avg.finish"


class ResNet50FeatureExtractor(nn.Module):
    """ResNet50 feature extractor with configurable Global Pooling (avg or max)."""
    def __init__(self, pooling="avg"):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Extract features up to the last convolutional layer (2048 channels)
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])

        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}. Use 'avg' or 'max'.")

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat)
        return torch.flatten(feat, 1)


class CroppedDiskDataset(Dataset):
    """Loads pre-cropped images from disk (Approach 2)."""
    def __init__(self, image_items, transform):
        self.image_items = image_items
        self.transform = transform

    def __len__(self):
        return len(self.image_items)

    def __getitem__(self, idx):
        path, key_name, mid, crop_i = self.image_items[idx]
        try:
            with Image.open(path) as img:
                tensor = self.transform(img.convert("RGB"))
                return tensor, key_name, mid, str(crop_i), True
        except Exception:
            return torch.zeros(3, 224, 224), key_name, mid, str(crop_i), False


class OnlineFiveCropDataset(Dataset):
    """Loads origin poster and creates 5 crops on-the-fly in RAM (Approach 1)."""
    def __init__(self, image_paths, five_crop_transform):
        self.image_paths = image_paths
        self.five_crop_transform = five_crop_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        stem = Path(path).stem
        try:
            with Image.open(path) as img:
                crops_tensor = self.five_crop_transform(img.convert("RGB"))  # Shape: (5, 3, 224, 224)
                return crops_tensor, stem, True
        except Exception:
            return torch.zeros(5, 3, 224, 224), stem, False


def get_single_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_five_crop_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([
            normalize(transforms.ToTensor()(crop)) for crop in crops
        ]))
    ])


def extract_from_disk_crops(model, dataloader, device, out_dir, finish_file, pooling_name, total_items):
    """Inference for pre-cropped disk images."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    finish_path = Path(finish_file)
    finish_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Extracting ResNet50 ({pooling_name.upper()}) from DISK CROPS ---")
    start_time = time.time()
    processed_count = 0
    pbar = tqdm(total=total_items, desc=f"ResNet50 Disk ({pooling_name})", unit="img") if tqdm else None
    finish_handle = open(finish_path, "a", encoding="utf-8")

    try:
        with torch.no_grad():
            for batch_imgs, batch_keys, batch_mids, batch_crops, batch_valids in dataloader:
                batch_imgs = batch_imgs.to(device, non_blocking=True)
                feats = model(batch_imgs).cpu().numpy()

                for i in range(len(batch_keys)):
                    if not batch_valids[i]:
                        continue
                    key = batch_keys[i]
                    mid = batch_mids[i]
                    crop_i = batch_crops[i]
                    feat = feats[i]

                    feat_file = out_path / f"{key}.ResNet50"
                    feat_str = ",".join([f"{v:.6f}" for v in feat])
                    with open(feat_file, "w", encoding="utf-8") as f:
                        f.write(f"{feat_str}\n")

                    if crop_i != "full":
                        finish_handle.write(f"{mid},{crop_i}\n")
                    else:
                        finish_handle.write(f"{mid}\n")

                    processed_count += 1

                if pbar:
                    pbar.update(len(batch_keys))
    finally:
        finish_handle.close()
        if pbar:
            pbar.close()

    elapsed = time.time() - start_time
    print(f"Finished {pooling_name}: {processed_count} images in {elapsed:.2f}s ({processed_count / max(elapsed, 0.001):.1f} imgs/s)")


def extract_online_five_crop(model, dataloader, device, out_dir, finish_file, pooling_name, total_movies):
    """Inference for online 5-crop directly in RAM/GPU without disk overhead (Approach 1)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    finish_path = Path(finish_file)
    finish_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Extracting ResNet50 ({pooling_name.upper()}) with ONLINE FIVE-CROP ---")
    start_time = time.time()
    processed_movies = 0
    pbar = tqdm(total=total_movies, desc=f"ResNet50 Online ({pooling_name})", unit="movie") if tqdm else None
    finish_handle = open(finish_path, "a", encoding="utf-8")

    # Corner index mapping for FiveCrop:
    # PyTorch FiveCrop order: 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right, 4: center
    # Original project mapping: 1: top-left, 2: top-right, 3: bottom-right, 4: bottom-left, 0: center
    pt_to_proj_crop = {0: 1, 1: 2, 2: 4, 3: 3, 4: 0}

    try:
        with torch.no_grad():
            for batch_crops, batch_stems, batch_valids in dataloader:
                # batch_crops shape: (B, 5, 3, 224, 224)
                B = batch_crops.shape[0]
                flat_crops = batch_crops.view(B * 5, 3, 224, 224).to(device, non_blocking=True)
                feats = model(flat_crops).cpu().numpy().reshape(B, 5, 2048)

                for b in range(B):
                    if not batch_valids[b]:
                        continue
                    stem = batch_stems[b]
                    movie_crops_feat = feats[b]

                    # Save all 5 individual crops: {id}_{i}.ResNet50
                    for pt_idx, proj_idx in pt_to_proj_crop.items():
                        crop_feat = movie_crops_feat[pt_idx]
                        feat_file = out_path / f"{stem}_{proj_idx}.ResNet50"
                        feat_str = ",".join([f"{v:.6f}" for v in crop_feat])
                        with open(feat_file, "w", encoding="utf-8") as f:
                            f.write(f"{feat_str}\n")
                        finish_handle.write(f"{stem},{proj_idx}\n")

                    # Also save the full combined movie feature: {id}.ResNet50 (mean over 5 crops)
                    full_feat = np.mean(movie_crops_feat, axis=0)
                    feat_file_full = out_path / f"{stem}.ResNet50"
                    feat_str_full = ",".join([f"{v:.6f}" for v in full_feat])
                    with open(feat_file_full, "w", encoding="utf-8") as f:
                        f.write(f"{feat_str_full}\n")
                    finish_handle.write(f"{stem}\n")

                    processed_movies += 1

                if pbar:
                    pbar.update(B)
    finally:
        finish_handle.close()
        if pbar:
            pbar.close()

    elapsed = time.time() - start_time
    total_crops = processed_movies * 5
    print(f"Finished {pooling_name}: {processed_movies} movies ({total_crops} crops) in {elapsed:.2f}s ({processed_movies / max(elapsed, 0.001):.1f} movies/s)")


def main():
    parser = argparse.ArgumentParser(description="High-Speed GPU ResNet50 Feature Extraction (Approaches 1 & 2)")
    parser.add_argument("--source", choices=["auto", "fivecrop", "crop"], default="auto",
                        help="'fivecrop'=Online 5-crop in RAM/GPU (Approach 1); 'crop'=from image_crop on disk (Approach 2); 'auto'=automatic")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for GPU inference (default: 128)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers (default: 8)")
    parser.add_argument("--pooling", choices=["avg", "max", "both"], default="both", help="Pooling layer type")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process (for testing)")
    parser.add_argument("--force", action="store_true", help="Re-extract even if features already exist")
    args = parser.parse_args()

    # GPU Check
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM)")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not detected, using CPU.")

    # Determine mode
    crop_dir = Path(PATH_IMAGE_CROP)
    origin_dir = Path(PATH_IMAGE_ORIGIN)

    mode = args.source
    if mode == "auto":
        origin_count = len(list(origin_dir.glob("*.jpg"))) if origin_dir.exists() else 0
        crop_count = len(list(crop_dir.glob("*.jpg"))) if crop_dir.exists() else 0
        if crop_count > 0 and crop_count >= origin_count * 4:
            mode = "crop"
        else:
            mode = "fivecrop"

    print(f"Selected Extraction Mode: [{mode.upper()}]")

    poolings_to_run = ["avg", "max"] if args.pooling == "both" else [args.pooling]

    if mode == "fivecrop":
        # Approach 1: Online FiveCrop in RAM/GPU directly from origin posters
        all_origin_images = sorted(list(origin_dir.glob("*.jpg")))
        if args.limit:
            all_origin_images = all_origin_images[:args.limit]

        print(f"Found {len(all_origin_images)} origin poster images in {origin_dir}.")

        for pool in poolings_to_run:
            out_dir = PATH_FEATURE_AVG if pool == "avg" else PATH_FEATURE_MAX
            finish_file = PATH_FEATURE_AVG_FINISH if pool == "avg" else PATH_FEATURE_MAX_FINISH

            # Resume check
            if not args.force:
                out_path = Path(out_dir)
                existing = {f.stem for f in out_path.glob("*.ResNet50")} if out_path.exists() else set()
                # Check if {stem}_0 exists
                filtered = [p for p in all_origin_images if f"{p.stem}_0" not in existing]
                print(f"[{pool.upper()}] {len(all_origin_images) - len(filtered)} movies already extracted. {len(filtered)} remaining.")
            else:
                filtered = all_origin_images

            if not filtered:
                print(f"[{pool.upper()}] All movies already processed.")
                continue

            # Batch size for Online FiveCrop: each movie is 5 images, so batch_size=32 means 160 images/forward
            effective_bs = max(1, args.batch_size // 4)
            dataset = OnlineFiveCropDataset(filtered, get_five_crop_transform())
            dataloader = DataLoader(
                dataset,
                batch_size=effective_bs,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=(device.type == "cuda")
            )

            model = ResNet50FeatureExtractor(pooling=pool).to(device)
            model.eval()

            extract_online_five_crop(
                model=model,
                dataloader=dataloader,
                device=device,
                out_dir=out_dir,
                finish_file=finish_file,
                pooling_name=pool,
                total_movies=len(filtered)
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    else:
        # Approach 2: From pre-cropped images on disk
        all_crop_images = sorted(list(crop_dir.glob("*.jpg")))
        if args.limit:
            all_crop_images = all_crop_images[:args.limit]

        image_items = []
        for f in all_crop_images:
            stem = f.stem
            mid, crop_i = stem.rsplit("_", 1) if "_" in stem else (stem, "0")
            image_items.append((str(f), stem, mid, crop_i))

        print(f"Found {len(image_items)} cropped images in {crop_dir}.")

        for pool in poolings_to_run:
            out_dir = PATH_FEATURE_AVG if pool == "avg" else PATH_FEATURE_MAX
            finish_file = PATH_FEATURE_AVG_FINISH if pool == "avg" else PATH_FEATURE_MAX_FINISH

            if not args.force:
                out_path = Path(out_dir)
                existing = {f.stem for f in out_path.glob("*.ResNet50")} if out_path.exists() else set()
                filtered = [item for item in image_items if item[1] not in existing]
                print(f"[{pool.upper()}] {len(existing)} cropped features already exist. {len(filtered)} remaining.")
            else:
                filtered = image_items

            if not filtered:
                print(f"[{pool.upper()}] All cropped features already extracted.")
                continue

            dataset = CroppedDiskDataset(filtered, get_single_transform())
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=(device.type == "cuda")
            )

            model = ResNet50FeatureExtractor(pooling=pool).to(device)
            model.eval()

            extract_from_disk_crops(
                model=model,
                dataloader=dataloader,
                device=device,
                out_dir=out_dir,
                finish_file=finish_file,
                pooling_name=pool,
                total_items=len(filtered)
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\nAll ResNet50 feature extractions completed successfully!")


if __name__ == "__main__":
    main()
