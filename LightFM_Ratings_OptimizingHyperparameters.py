import os
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
import support_metric as sp
import support_dataset as spDataset

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")
DEFAULT_MIN_USER_RATING = int(os.environ.get("MIN_USER_RATING", "5"))


def sample_hyperparameters():
    while True:
        yield {
            "no_components": np.random.randint(16, 64),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp"]),
            "learning_rate": float(np.random.exponential(0.05)),
            "item_alpha": float(np.random.exponential(1e-3)),
            "user_alpha": float(np.random.exponential(1e-3)),
            "max_sampled": int(np.random.randint(5, 25)),
            "num_epochs": int(np.random.randint(30, 80)),
        }


def random_search(train, test, num_samples=50, num_threads=8, log_file=None):
    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")
        model = LightFM(**hyperparams)
        model.fit(train, epochs=num_epochs, num_threads=num_threads)
        score = sp.precision_at_k(model, test, k=1)
        hyperparams["num_epochs"] = num_epochs
        print("Score {:.4f} at {}".format(score, hyperparams))
        if log_file:
            with open(log_file, "a") as out_hyper:
                out_hyper.write("Score {:.4f} at {}\n".format(score, hyperparams))
        yield (score, hyperparams, model)


def main():
    parser = argparse.ArgumentParser(description="Optimize Hyperparameters for LightFM Ratings on MovieLens")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--min-user-rating", type=int, default=DEFAULT_MIN_USER_RATING, help="Min user ratings threshold")
    parser.add_argument("--samples", type=int, default=30, help="Number of random search samples")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 8), help="Threads")
    args = parser.parse_args()

    ml_size = args.size
    min_user_rating = args.min_user_rating

    Path("dataset").mkdir(parents=True, exist_ok=True)
    Path("hyperparams").mkdir(parents=True, exist_ok=True)

    path_train_dataset = f"dataset/train_ml-{ml_size}_UMR({min_user_rating}).pd"
    path_test_dataset = f"dataset/test_ml-{ml_size}_UMR({min_user_rating}).pd"
    log_file = f"hyperparams/lightfm_ratings_ml-{ml_size}.txt"

    print(f"Loading MovieLens [{ml_size}] with min_user_ratings={min_user_rating}...")
    ratings = spDataset.load_movielens_data(size=ml_size, min_user_ratings=min_user_rating)

    if os.path.exists(path_train_dataset) and os.path.exists(path_test_dataset):
        ratings_train = pd.read_pickle(path_train_dataset)
        ratings_test = pd.read_pickle(path_test_dataset)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset(ratings)
        ratings_train.to_pickle(path_train_dataset)
        ratings_test.to_pickle(path_test_dataset)

    dataset = Dataset()
    dataset.fit(ratings['userID'].unique(), ratings['itemID'].unique())

    (interactions_train, weights_train) = dataset.build_interactions(
        (x['userID'], x['itemID'], x['rating']) for _, x in ratings_train.iterrows()
    )
    (interactions_test, weights_test) = dataset.build_interactions(
        (x['userID'], x['itemID'], x['rating']) for _, x in ratings_test.iterrows()
    )

    train = spDataset.build_positive_data(interactions_train, weights_train)
    test = spDataset.build_positive_data(interactions_test, weights_test)

    best_score, best_hyperparams, best_model = max(
        random_search(train, test, num_samples=args.samples, num_threads=args.threads, log_file=log_file),
        key=lambda x: x[0]
    )

    print("=" * 60)
    print(f"Best score {best_score:.4f} at {best_hyperparams}")
    with open(log_file, "a") as out_hyper:
        out_hyper.write(f"** BEST: Score {best_score:.4f} at {best_hyperparams}\n")


if __name__ == "__main__":
    main()