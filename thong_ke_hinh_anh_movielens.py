import os
import argparse
import pandas as pd
import support_dataset as spDataset

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")


def main():
    parser = argparse.ArgumentParser(description="Statistics of poster images for MovieLens")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--image-dir", default="image_orgin/movielens", help="Directory containing poster images")
    args = parser.parse_args()

    ml_size = args.size
    image_dir = args.image_dir

    print(f"Loading MovieLens [{ml_size}]...")
    df = spDataset.load_movielens_data(size=ml_size)
    unique_items = df['itemID'].unique()
    print(f"Total unique items in ratings: {len(unique_items)}")

    has_image = 0
    missing_image = 0
    missing_ids = []

    for iid in unique_items:
        path_image = os.path.join(image_dir, f"{iid}.jpg")
        if os.path.exists(path_image):
            has_image += 1
        else:
            missing_image += 1
            missing_ids.append(iid)

    print(f"Items with image:    {has_image} ({has_image / len(unique_items) * 100:.2f}%)")
    print(f"Items without image: {missing_image} ({missing_image / len(unique_items) * 100:.2f}%)")


if __name__ == "__main__":
    main()