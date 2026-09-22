import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import support_dataset as spDataset

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")


def main():
    parser = argparse.ArgumentParser(description="Plot rating histogram for MovieLens")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    args = parser.parse_args()

    Path("hist").mkdir(parents=True, exist_ok=True)
    df = spDataset.load_movielens_data(size=args.size)
    plt.figure()
    df["rating"].hist()
    plt.title(f"Rating Distribution (ml-{args.size})")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    out_fig = f"hist/movielens-{args.size}.png"
    plt.savefig(out_fig)
    plt.close()
    print(f"Histogram saved to {out_fig}")


if __name__ == "__main__":
    main()