import os
import argparse
import pickle
import pandas as pd
from pathlib import Path
import support_dataset as spDataset

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")
DEFAULT_MIN_USER_RATING = int(os.environ.get("MIN_USER_RATING", "5"))


def main():
    parser = argparse.ArgumentParser(description="Analyze Train/Test Distribution for MovieLens")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--min-user-rating", type=int, default=DEFAULT_MIN_USER_RATING, help="Min user ratings threshold (default: 5)")
    args = parser.parse_args()

    ml_size = args.size
    min_user_rating = args.min_user_rating
    path_folder_dataset = "dataset"
    Path(path_folder_dataset).mkdir(parents=True, exist_ok=True)

    path_train_dataset = f"{path_folder_dataset}/train_ml-{ml_size}_UMR({min_user_rating}).pd"
    path_test_dataset = f"{path_folder_dataset}/test_ml-{ml_size}_UMR({min_user_rating}).pd"

    print(f"Loading MovieLens [{ml_size}] with min_user_ratings={min_user_rating}...")
    ratings = spDataset.load_movielens_data(size=ml_size, min_user_ratings=min_user_rating)

    if os.path.exists(path_train_dataset) and os.path.exists(path_test_dataset):
        print(f"Loading cached splits from {path_folder_dataset}...")
        ratings_train = pd.read_pickle(path_train_dataset)
        ratings_test = pd.read_pickle(path_test_dataset)
    else:
        print("Creating new train/test split (75/25)...")
        ratings_train, ratings_test = spDataset.get_train_test_dataset(ratings)
        ratings_train.to_pickle(path_train_dataset)
        ratings_test.to_pickle(path_test_dataset)

    print(f"Ratings shape: {ratings.shape}")
    print(f"Train shape: {ratings_train.shape}")
    user_ratings_train = ratings_train.groupby(['userID'])['itemID'].count().reset_index().sort_values(["itemID"])
    print(f"Train - Mean ratings/user: {user_ratings_train['itemID'].mean():.2f}")
    print(f"Train - Min ratings/user:  {user_ratings_train['itemID'].min()}")
    print(f"Train - Max ratings/user:  {user_ratings_train['itemID'].max()}")

    print(f"Test shape: {ratings_test.shape}")
    user_ratings_test = ratings_test.groupby(['userID'])['itemID'].count().reset_index().sort_values(["itemID"])
    print(f"Test - Mean ratings/user:  {user_ratings_test['itemID'].mean():.2f}")
    print(f"Test - Min ratings/user:   {user_ratings_test['itemID'].min()}")
    print(f"Test - Max ratings/user:   {user_ratings_test['itemID'].max()}")


if __name__ == "__main__":
    main()