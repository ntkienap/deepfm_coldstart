import os
import sys
import time
import argparse
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
import cv2
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_DATA = os.environ.get("PATH_ROOT_DATA", str(BASE_DIR))


def compute_sift_from_gray(gray_np):
    """Compute SIFT descriptors for a grayscale image numpy array (uint8).
    Creates a new SIFT instance per call to ensure 100% thread/process safety.
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray_np, None)
    if des is None or len(des) == 0:
        return None
    return des.astype(np.float32)


def descriptors_to_bow(des, centers, num_clusters):
    """Vectorized calculation of visual word histogram using squared Euclidean distance."""
    if des is None or len(des) == 0:
        return np.zeros(num_clusters, dtype=np.float32)

    # cdist in C/SciPy is much faster than running KMeans.predict() in python loops
    dists = cdist(des, centers, metric="sqeuclidean")
    cluster_ids = np.argmin(dists, axis=1)
    hist = np.bincount(cluster_ids, minlength=num_clusters).astype(np.float32)
    total = np.sum(hist)
    if total > 0:
        hist /= total
    return hist


# --- WORKER FUNCTIONS FOR MULTIPROCESSING ---

def worker_sample_sift(args):
    """Worker function to extract SIFT descriptors for codebook training."""
    img_path, target_size = args
    try:
        with Image.open(img_path) as img:
            gray = img.convert("L").resize(target_size, Image.Resampling.BILINEAR)
            gray_np = np.array(gray, dtype=np.uint8)
            return compute_sift_from_gray(gray_np)
    except Exception:
        return None


def worker_extract_fivecrop(args):
    """Worker function for Approach 1: Online Five-Crop in RAM."""
    img_path, centers, num_clusters, out_dir, force, size_image = args
    stem = Path(img_path).stem
    out_path = Path(out_dir)

    # PyTorch/Project index mapping: 0=center, 1=top-left, 2=top-right, 3=bottom-right, 4=bottom-left
    crop_files = [out_path / f"{stem}_{i}.bow" for i in range(5)]
    full_file = out_path / f"{stem}.bow"

    if not force:
        if all(cf.exists() and cf.stat().st_size > 10 for cf in crop_files) and full_file.exists():
            return stem, "skipped"

    try:
        with Image.open(img_path) as img:
            img_gray = img.convert("L")
            w, h = img_gray.size
            min_size = min(w, h)
            size_result = size_image + 0.33 * size_image
            w_result = round(w / min_size * size_result)
            h_result = round(h / min_size * size_result)

            resized = img_gray.resize((w_result, h_result), Image.Resampling.BILINEAR)

            # 4 Corners + 1 Center
            crops = {
                1: resized.crop((0, 0, size_image, size_image)),
                2: resized.crop((w_result - size_image, 0, w_result, size_image)),
                3: resized.crop((w_result - size_image, h_result - size_image, w_result, h_result)),
                4: resized.crop((0, h_result - size_image, size_image, h_result)),
            }
            left = round((w_result - size_image) / 2)
            top = round((h_result - size_image) / 2)
            crops[0] = resized.crop((left, top, left + size_image, top + size_image))

            bow_vectors = []
            for crop_idx in range(5):
                crop_np = np.array(crops[crop_idx], dtype=np.uint8)
                des = compute_sift_from_gray(crop_np)
                hist = descriptors_to_bow(des, centers, num_clusters)
                bow_vectors.append(hist)

                feat_str = ",".join([f"{v:.6f}" for v in hist])
                with open(crop_files[crop_idx], "w", encoding="utf-8") as f:
                    f.write(f"{feat_str}\n")

            # Combined movie feature: mean over 5 crops
            full_bow = np.mean(bow_vectors, axis=0)
            feat_str_full = ",".join([f"{v:.6f}" for v in full_bow])
            with open(full_file, "w", encoding="utf-8") as f:
                f.write(f"{feat_str_full}\n")

        return stem, "success"
    except Exception:
        return stem, "error"


def worker_extract_single(args):
    """Worker function for Approaches 2 & 3: Single file per item."""
    img_path, centers, num_clusters, out_dir, force, target_size = args
    stem = Path(img_path).stem
    out_file = Path(out_dir) / f"{stem}.bow"

    if not force and out_file.exists() and out_file.stat().st_size > 10:
        return stem, "skipped"

    try:
        with Image.open(img_path) as img:
            gray = img.convert("L")
            if target_size:
                gray = gray.resize(target_size, Image.Resampling.BILINEAR)
            gray_np = np.array(gray, dtype=np.uint8)
            des = compute_sift_from_gray(gray_np)
            hist = descriptors_to_bow(des, centers, num_clusters)

            feat_str = ",".join([f"{v:.6f}" for v in hist])
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"{feat_str}\n")

        return stem, "success"
    except Exception:
        return stem, "error"


# --- TRAINING CODEBOOK ---

def get_or_train_bow_model(sample_images, model_path, num_clusters, workers, sample_size=1000):
    """Trains MiniBatchKMeans visual vocabulary if model does not exist yet."""
    model_file = Path(model_path)
    if model_file.exists():
        print(f"Loading pre-trained BoW codebook from {model_file}...")
        with open(model_file, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict) and "centers" in data:
                return data["centers"]
            elif hasattr(data, "cluster_centers_"):
                return data.cluster_centers_
            else:
                return data

    model_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Training BoW Visual Vocabulary ({num_clusters} clusters) ---")
    np.random.seed(42)
    selected_samples = list(sample_images)
    if len(selected_samples) > sample_size:
        indices = np.random.choice(len(selected_samples), sample_size, replace=False)
        selected_samples = [selected_samples[i] for i in indices]

    print(f"Sampling SIFT descriptors from {len(selected_samples)} images using {workers} CPU workers...")
    task_args = [(str(p), (256, 256)) for p in selected_samples]

    collected_des = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for des in executor.map(worker_sample_sift, task_args, chunksize=16):
            if des is not None:
                # Subsample descriptors per image to keep memory bounded
                if len(des) > 300:
                    sub_idx = np.random.choice(len(des), 300, replace=False)
                    collected_des.append(des[sub_idx])
                else:
                    collected_des.append(des)

    if not collected_des:
        raise RuntimeError("Failed to extract any SIFT descriptors for BoW codebook training.")

    all_descriptors = np.vstack(collected_des)
    print(f"Collected {len(all_descriptors)} SIFT descriptors. Training MiniBatchKMeans...")

    start_train = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=2048,
        random_state=42,
        max_iter=100,
        verbose=0
    ).fit(all_descriptors)
    print(f"KMeans training finished in {time.time() - start_train:.2f}s.")

    centers = kmeans.cluster_centers_
    with open(model_file, "wb") as f:
        pickle.dump({"kmeans": kmeans, "centers": centers}, f)
    print(f"Saved BoW dictionary model to {model_file}.")

    return centers


def main():
    parser = argparse.ArgumentParser(description="High-Speed Multi-Core SIFT + BoW (Bag-of-Visual-Words) Feature Extraction")
    parser.add_argument("--dataset", choices=["movielens", "movietweeting"], default="movielens",
                        help="Target dataset (default: movielens)")
    parser.add_argument("--source", choices=["auto", "fivecrop", "crop", "resize"], default="auto",
                        help="'fivecrop'=Online 5-crop in RAM; 'crop'=from image_crop; 'resize'=from image_resize; 'auto'=automatic")
    parser.add_argument("--clusters", type=int, default=64, help="Number of visual words / clusters (default: 64)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8), help="CPU worker processes (default: 16)")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of images sampled for codebook training (default: 1000)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process (for testing)")
    parser.add_argument("--force", action="store_true", help="Re-extract even if features already exist")
    args = parser.parse_args()

    root_data = Path(DEFAULT_ROOT_DATA).resolve()
    dataset_name = "movielens" if args.dataset == "movielens" else "MovieTweeting"

    origin_dir = root_data / "image_orgin" / args.dataset
    if not origin_dir.exists() and (root_data / "image_orgin" / "movielens").exists() and args.dataset == "movielens":
        origin_dir = root_data / "image_orgin" / "movielens"

    crop_dir = root_data / "image_crop" / args.dataset
    resize_dir = root_data / "image_resize" / args.dataset

    out_bow_dir = root_data / "feature" / dataset_name / f"BoW_{args.clusters}"
    out_bow_dir.mkdir(parents=True, exist_ok=True)
    finish_file = root_data / "feature" / dataset_name / f"BoW_{args.clusters}.finish"
    finish_file.parent.mkdir(parents=True, exist_ok=True)
    model_path = root_data / "model" / f"bow_dictionary_{args.clusters}_{args.dataset}.pkl"

    # Determine mode
    mode = args.source
    if mode == "auto":
        origin_count = len(list(origin_dir.glob("*.jpg"))) if origin_dir.exists() else 0
        crop_count = len(list(crop_dir.glob("*.jpg"))) if crop_dir.exists() else 0
        if crop_count > 0 and crop_count >= origin_count * 4:
            mode = "crop"
        else:
            mode = "fivecrop"

    print(f"Selected Dataset: [{dataset_name}] | Mode: [{mode.upper()}] | Clusters: [{args.clusters}] | Workers: [{args.workers}]")

    # 1. Train or load codebook centers
    sample_pool = []
    if origin_dir.exists():
        sample_pool.extend(list(origin_dir.glob("*.jpg")))
    elif crop_dir.exists():
        sample_pool.extend(list(crop_dir.glob("*.jpg")))

    if not sample_pool:
        print(f"Error: No images found in {origin_dir} or {crop_dir} to train or extract features.", file=sys.stderr)
        sys.exit(1)

    centers = get_or_train_bow_model(
        sample_images=sample_pool,
        model_path=model_path,
        num_clusters=args.clusters,
        workers=args.workers,
        sample_size=args.sample_size
    )

    # 2. Prepare tasks based on mode
    start_time = time.time()
    finish_handle = open(finish_file, "a", encoding="utf-8")

    try:
        if mode == "fivecrop":
            all_origin = sorted(list(origin_dir.glob("*.jpg")))
            if args.limit:
                all_origin = all_origin[:args.limit]

            print(f"Processing {len(all_origin)} movies with ONLINE FIVECROP...")
            task_args = [
                (str(p), centers, args.clusters, str(out_bow_dir), args.force, 224)
                for p in all_origin
            ]

            pbar = tqdm(total=len(task_args), desc=f"BoW_{args.clusters} FiveCrop", unit="movie") if tqdm else None
            processed_count = 0

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for stem, status in executor.map(worker_extract_fivecrop, task_args, chunksize=16):
                    if status != "error":
                        for c_i in range(5):
                            finish_handle.write(f"{stem},{c_i}\n")
                        finish_handle.write(f"{stem}\n")
                        processed_count += 1
                    if pbar:
                        pbar.update(1)

            if pbar:
                pbar.close()

        elif mode == "crop":
            all_crop = sorted(list(crop_dir.glob("*.jpg")))
            if args.limit:
                all_crop = all_crop[:args.limit]

            print(f"Processing {len(all_crop)} cropped images from disk...")
            task_args = [
                (str(p), centers, args.clusters, str(out_bow_dir), args.force, None)
                for p in all_crop
            ]

            pbar = tqdm(total=len(task_args), desc=f"BoW_{args.clusters} Disk Crop", unit="img") if tqdm else None
            processed_count = 0

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for stem, status in executor.map(worker_extract_single, task_args, chunksize=16):
                    if status != "error":
                        if "_" in stem:
                            mid, c_i = stem.rsplit("_", 1)
                            finish_handle.write(f"{mid},{c_i}\n")
                        else:
                            finish_handle.write(f"{stem}\n")
                        processed_count += 1
                    if pbar:
                        pbar.update(1)

            if pbar:
                pbar.close()

        elif mode == "resize":
            source_dir = resize_dir if (resize_dir.exists() and any(resize_dir.glob("*.jpg"))) else origin_dir
            all_resize = sorted(list(source_dir.glob("*.jpg")))
            if args.limit:
                all_resize = all_resize[:args.limit]

            print(f"Processing {len(all_resize)} single resized images from {source_dir}...")
            task_args = [
                (str(p), centers, args.clusters, str(out_bow_dir), args.force, (224, 224))
                for p in all_resize
            ]

            pbar = tqdm(total=len(task_args), desc=f"BoW_{args.clusters} Resize", unit="img") if tqdm else None
            processed_count = 0

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for stem, status in executor.map(worker_extract_single, task_args, chunksize=16):
                    if status != "error":
                        finish_handle.write(f"{stem}\n")
                        processed_count += 1
                    if pbar:
                        pbar.update(1)

            if pbar:
                pbar.close()

    finally:
        finish_handle.close()

    elapsed = time.time() - start_time
    print(f"\nAll BoW ({args.clusters}-dim) extractions completed in {elapsed:.2f}s! Output saved to: {out_bow_dir}")


if __name__ == "__main__":
    main()
