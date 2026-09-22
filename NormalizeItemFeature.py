import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn.preprocessing
import support_dataset as spDataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = str(BASE_DIR) if (BASE_DIR / "movielens").exists() else "/home/nckh/nckh_code"
PATH_ROOT_DATA = os.environ.get("PATH_ROOT_DATA", DEFAULT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="L1 Normalization for Item Features")
    parser.add_argument("--size", default=os.environ.get("MOVIELENS_SIZE", "25m"),
                        help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--feature", choices=["VGG19", "ResNet50", "BoW"], default="VGG19",
                        help="Feature type (default: VGG19)")
    parser.add_argument("--pooling", choices=["avg", "max"], default="avg",
                        help="Pooling type (default: avg)")
    parser.add_argument("--clusters", type=int, default=64,
                        help="Number of clusters for BoW (default: 64)")
    args = parser.parse_args()

    ml_info = spDataset.get_movielens_paths(args.size)
    movies_path = ml_info["movies_path"]
    sep = ml_info["sep"]
    has_header = ml_info["has_header"]

    if not os.path.exists(movies_path):
        print(f"Error: Movies file not found at {movies_path}", file=sys.stderr)
        sys.exit(1)

    if args.feature == "BoW":
        feat_dir_name = f"BoW_{args.clusters}"
        ext_in = "bow"
        ext_out = f"BoW_{args.clusters}_l1"
    else:
        feat_dir_name = f"{args.feature}_{args.pooling}"
        ext_in = args.feature
        ext_out = f"{args.feature}_l1"

    path_feature_in = Path(PATH_ROOT_DATA) / "feature" / "movielens" / feat_dir_name
    path_feature_out = Path(PATH_ROOT_DATA) / "feature" / "movielens" / f"{feat_dir_name}_L1"
    path_feature_out.mkdir(parents=True, exist_ok=True)

    print(f"Normalizing features from: {path_feature_in}")
    print(f"Output directory: {path_feature_out}")

    if has_header:
        movies_df = pd.read_csv(movies_path, sep=sep, engine="python", header=0)
        items_set = movies_df.iloc[:, 0].drop_duplicates().sort_values().values
    else:
        names = ["items", "title", "genres"]
        movies_df = pd.read_csv(movies_path, sep=sep, names=names, engine="python", encoding="utf-8")
        items_set = movies_df["items"].drop_duplicates().sort_values().values

    print(f"Found {len(items_set)} movie items from {movies_path}.")

    features = []
    labels = []

    pbar = tqdm(total=len(items_set), desc="Loading feature files", unit="item") if tqdm else None
    for id_item in items_set:
        for i in range(5):
            f_path = path_feature_in / f"{id_item}_{i}.{ext_in}"
            if f_path.exists() and f_path.stat().st_size > 10:
                try:
                    feat = np.loadtxt(f_path, delimiter=",", dtype=np.float64)
                    features.append(feat)
                    labels.append({"id": id_item, "i": i})
                except Exception:
                    pass

        f_path_full = path_feature_in / f"{id_item}.{ext_in}"
        if f_path_full.exists() and f_path_full.stat().st_size > 10:
            try:
                feat = np.loadtxt(f_path_full, delimiter=",", dtype=np.float64)
                features.append(feat)
                labels.append({"id": id_item, "i": None})
            except Exception:
                pass

        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    if not features:
        print(f"No features found to normalize in {path_feature_in}.")
        return

    print(f"Normalizing {len(features)} feature vectors with L1 norm...")
    norm_features = sklearn.preprocessing.normalize(np.array(features), norm="l1", copy=False)

    print(f"Writing normalized features to {path_feature_out}...")
    pbar_write = tqdm(total=len(labels), desc="Saving normalized features", unit="file") if tqdm else None
    for idx, meta in enumerate(labels):
        f_vec = norm_features[idx]
        f_str = ",".join([f"{v:.6f}" for v in f_vec])
        if meta["i"] is None:
            out_file = path_feature_out / f"{meta['id']}.{ext_out}"
        else:
            out_file = path_feature_out / f"{meta['id']}_{meta['i']}.{ext_out}"

        with open(out_file, "w", encoding="utf-8") as out_f:
            out_f.write(f"{f_str}\n")

        if pbar_write:
            pbar_write.update(1)

    if pbar_write:
        pbar_write.close()

    print(f"L1 Normalization completed successfully! Saved {len(labels)} files.")


if __name__ == "__main__":
    main()
