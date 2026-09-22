import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import support_dataset as spDataset

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")
DEFAULT_MIN_USER_RATING = int(os.environ.get("MIN_USER_RATING", "50"))
DEFAULT_MIN_MOVIE_RATING = int(os.environ.get("MIN_MOVIE_RATING", "5"))


def main():
    parser = argparse.ArgumentParser(description="Plot Ratings Distribution for MovieLens")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--min-user-rating", type=int, default=DEFAULT_MIN_USER_RATING, help="Min user ratings threshold (default: 50)")
    parser.add_argument("--min-movie-rating", type=int, default=DEFAULT_MIN_MOVIE_RATING, help="Min movie ratings threshold (default: 5)")
    args = parser.parse_args()

    ml_size = args.size
    Path("hist").mkdir(parents=True, exist_ok=True)

    print(f"Loading MovieLens [{ml_size}] (min_user={args.min_user_rating}, min_movie={args.min_movie_rating})...")
    df = spDataset.load_movielens_data(
        size=ml_size,
        min_user_ratings=args.min_user_rating,
        min_movie_ratings=args.min_movie_rating
    )

    user_ratings = df.groupby(['userID'])['itemID'].count().reset_index().sort_values(["itemID"])
    item_ratings = df.groupby(['itemID'])['userID'].count().reset_index().sort_values(["userID"])

    plt.figure()
    sns.barplot(data=item_ratings, x="itemID", y="userID")
    plt.ylabel("Số lượt đánh giá")
    plt.title(f"SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI ITEM (ml-{ml_size})")
    item_fig = f"hist/item_ratings_ml-{ml_size}_filter.png"
    plt.savefig(item_fig)
    plt.close()
    print(f"Saved: {item_fig}")

    plt.figure()
    sns.barplot(data=user_ratings, x="userID", y="itemID")
    plt.ylabel("Số lượt đánh giá")
    plt.title(f"SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI USER (ml-{ml_size})")
    user_fig = f"hist/user_ratings_ml-{ml_size}_filter.png"
    plt.savefig(user_fig)
    plt.close()
    print(f"Saved: {user_fig}")


if __name__ == "__main__":
    main()
