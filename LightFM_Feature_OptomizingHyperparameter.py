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
DEFAULT_FEATURE_NAME = os.environ.get("FEATURE_NAME", "VGG19")
DEFAULT_POOLING = os.environ.get("POOLING", "avg")


def sample_hyperparameters():
    while True:
        yield {
            "no_components": int(np.random.randint(8, 32)),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp"]),
            "learning_rate": round(float(np.random.uniform(1e-3, 1e-5)), 6),
            "item_alpha": round(float(np.random.uniform(1e-3, 1e-5)), 6),
            "user_alpha": round(float(np.random.uniform(1e-3, 1e-5)), 6),
            "max_sampled": int(np.random.randint(5, 25)),
            "num_epochs": 10,
        }


def random_search(dataset, ratings_train, train, test, feature_test, size_vector=512, pooling="avg", feature_name="VGG19", num_samples=30, num_threads=8, log_file=None):
    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        try:
            num_epochs = hyperparams.pop("num_epochs")
            model = LightFM(**hyperparams)
            for _ in range(num_epochs):
                item_matching_train = spDataset.get_image_features(
                    ratings_train, size_vector=size_vector, pooling=pooling, feature_name=feature_name
                )
                item_features_train = dataset.build_item_features(
                    ((x[0], x[1]) for x in item_matching_train), normalize=False
                )
                model.fit_partial(train, epochs=1, item_features=item_features_train, num_threads=num_threads)

            score = sp.full_auc(model, test, feature=feature_test)
            hyperparams["num_epochs"] = num_epochs
            print(f"Score {score:.4f} at {hyperparams}")
            if log_file:
                with open(log_file, "a") as out_hyper:
                    out_hyper.write(f"Score {score:.4f} at {hyperparams}\n")
            yield (score, hyperparams, model)
        except Exception as e:
            print(f"Error in trial: {e}")


def main():
    parser = argparse.ArgumentParser(description="Optimize Hyperparameters for LightFM with Features on MovieLens")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--min-user-rating", type=int, default=DEFAULT_MIN_USER_RATING, help="Min user ratings threshold")
    parser.add_argument("--feature", default=DEFAULT_FEATURE_NAME, help="Feature name (default: VGG19)")
    parser.add_argument("--pooling", default=DEFAULT_POOLING, choices=["avg", "max"], help="Pooling (default: avg)")
    parser.add_argument("--samples", type=int, default=20, help="Number of random search samples")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 8), help="Threads")
    args = parser.parse_args()

    ml_size = args.size
    min_user_rating = args.min_user_rating
    feature_name = args.feature

    size_vector = 2048 if "RESNET" in feature_name.upper() else (512 if "VGG" in feature_name.upper() else 64)

    Path("dataset").mkdir(parents=True, exist_ok=True)
    Path("hyperparams").mkdir(parents=True, exist_ok=True)

    path_train_dataset = f"dataset/train_ml-{ml_size}_UMR({min_user_rating}).pd"
    path_test_dataset = f"dataset/test_ml-{ml_size}_UMR({min_user_rating}).pd"
    log_file = f"hyperparams/lightfm_feature({feature_name})_ml-{ml_size}.txt"

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
    dataset.fit(
        ratings['userID'].unique(),
        ratings['itemID'].unique(),
        item_features=[f"f{i+1}" for i in range(size_vector)]
    )

    (interactions_train, _) = dataset.build_interactions(
        (x['userID'], x['itemID'], x['rating']) for _, x in ratings_train.iterrows()
    )
    (interactions_test, _) = dataset.build_interactions(
        (x['userID'], x['itemID'], x['rating']) for _, x in ratings_test.iterrows()
    )

    item_matching_test = spDataset.get_image_features(
        ratings, isTest=True, size_vector=size_vector, pooling=args.pooling, feature_name=feature_name
    )
    item_features_test = dataset.build_item_features(
        ((x[0], x[1]) for x in item_matching_test), normalize=False
    )

    with open(log_file, "w") as out_hyper:
        out_hyper.write(f"Hyperparameter tuning for {feature_name} on ml-{ml_size}\n")

    best_score, best_hyperparams, best_model = max(
        random_search(
            dataset, ratings_train, interactions_train, interactions_test, item_features_test,
            size_vector=size_vector, pooling=args.pooling, feature_name=feature_name,
            num_samples=args.samples, num_threads=args.threads, log_file=log_file
        ),
        key=lambda x: x[0]
    )

    print("=" * 60)
    print(f"Best score {best_score:.4f} at {best_hyperparams}")
    with open(log_file, "a") as out_hyper:
        out_hyper.write(f"** BEST: Score {best_score:.4f} at {best_hyperparams}\n")


if __name__ == "__main__":
    main()
