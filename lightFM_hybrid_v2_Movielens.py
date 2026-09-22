import os
import sys
import pickle
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import auc_score
import support_metric as sp
import support_dataset as spDataset
import matplotlib.pyplot as plt
import seaborn as sns
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")
DEFAULT_MIN_USER_RATING = int(os.environ.get("MIN_USER_RATING", "5"))
DEFAULT_FEATURE_NAME = os.environ.get("FEATURE_NAME", "VGG19")
DEFAULT_POOLING = os.environ.get("POOLING", "avg")


def parse_args():
    parser = argparse.ArgumentParser(description="LightFM Hybrid (Ratings + Visual Features) on MovieLens")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--min-user-rating", type=int, default=DEFAULT_MIN_USER_RATING, help="Min user ratings threshold (default: 5)")
    parser.add_argument("--feature", default=DEFAULT_FEATURE_NAME, choices=["VGG19", "ResNet50", "SIFT", "BoW"], help="Feature name (default: VGG19)")
    parser.add_argument("--size-vector", type=int, default=None, help="Vector dimension (auto-deduced if not specified)")
    parser.add_argument("--pooling", default=DEFAULT_POOLING, choices=["avg", "max"], help="Pooling method (default: avg)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 8), help="LightFM worker threads")
    return parser.parse_args()


def main():
    args = parse_args()
    ml_size = args.size
    min_user_rating = args.min_user_rating
    feature_name = args.feature
    pooling = args.pooling

    if args.size_vector is not None:
        size_vector = args.size_vector
    else:
        if "RESNET" in feature_name.upper():
            size_vector = 2048
        elif "VGG" in feature_name.upper():
            size_vector = 512
        else:
            size_vector = 64

    path_folder_dataset = "dataset"
    path_file_lightfm_model = f"model/lightfm_with_feature({feature_name})_ml-{ml_size}_UMR({min_user_rating}).model"
    path_train_dataset = f"{path_folder_dataset}/train_ml-{ml_size}_UMR({min_user_rating}).pd"
    path_test_dataset = f"{path_folder_dataset}/test_ml-{ml_size}_UMR({min_user_rating}).pd"

    for d in [path_folder_dataset, "model", "prediction_all", "result_evaluation", "evaluation"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    env = {
        "FEATURE_NAME": feature_name,
        "MOVIELENS_SIZE": ml_size,
        "MIN_USER_RATING": min_user_rating,
        "isNew": False
    }

    print(f"Loading MovieLens [{ml_size}] (min_user_rating={min_user_rating}, feature={feature_name}, dim={size_vector}, pooling={pooling})...")
    ratings = spDataset.load_movielens_data(size=ml_size, min_user_ratings=min_user_rating)

    if os.path.exists(path_train_dataset) and os.path.exists(path_test_dataset):
        print(f"Loading cached train/test splits from {path_folder_dataset}...")
        ratings_train = pd.read_pickle(path_train_dataset)
        ratings_test = pd.read_pickle(path_test_dataset)
    else:
        print("Creating new train/test split (75/25)...")
        ratings_train, ratings_test = spDataset.get_train_test_dataset(ratings)
        ratings_train.to_pickle(path_train_dataset)
        ratings_test.to_pickle(path_test_dataset)

    dataset = Dataset()
    dataset.fit(
        ratings['userID'].unique(),
        ratings['itemID'].unique(),
        item_features=[f"f{i+1}" for i in range(size_vector)]
    )

    num_users, num_items = dataset.interactions_shape()
    print(f"Num users: {num_users}, Num items: {num_items}.")

    print("Building interactions...")
    interactions_train, _ = dataset.build_interactions(
        (x['userID'], x['itemID'], x['rating']) for _, x in ratings_train.iterrows()
    )

    print("Building visual features for train items...")
    item_matching_train = spDataset.get_image_features(
        ratings_train, isTest=True, size_vector=size_vector, pooling=pooling, feature_name=feature_name
    )
    item_features_train = dataset.build_item_features(
        ((x[0], x[1]) for x in item_matching_train), normalize=True
    )

    print("Building visual features for test items...")
    item_matching_test = spDataset.get_image_features(
        ratings, isTest=True, size_vector=size_vector, pooling=pooling, feature_name=feature_name
    )
    item_features_test = dataset.build_item_features(
        ((x[0], x[1]) for x in item_matching_test), normalize=True
    )

    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()

    if os.path.exists(path_file_lightfm_model):
        print(f"Loading existing model from {path_file_lightfm_model}...")
        with open(path_file_lightfm_model, 'rb') as f:
            model = pickle.load(f)
    else:
        print(f"Training LightFM Hybrid model ({args.epochs} epochs, {args.threads} threads)...")
        model = LightFM(loss='warp', item_alpha=0.00001)
        model.fit_partial(
            interactions_train,
            item_features=item_features_train,
            epochs=args.epochs,
            verbose=True,
            num_threads=args.threads
        )
        with open(path_file_lightfm_model, 'wb') as f:
            pickle.dump(model, f)

    try:
        auc_val = auc_score(model, interactions_train, item_features=item_features_train, num_threads=args.threads).mean()
        print(f"Train Mean AUC: {auc_val:.4f}")
    except Exception as e:
        print(f"Could not compute train AUC: {e}")

    print("Predicting and evaluating Precision@k, Recall@k, F1@k...")
    evaluation = []
    df_test = ratings_test[['userID', 'itemID', 'rating']].copy()

    for k in range(1, 21):
        print(f"Computing metrics for k={k}...")
        eval_res = sp.precision_recall_f1_at_k(
            ratings, user_id_mapping, item_id_mapping,
            df_test=df_test,
            model=model, env=env,
            item_features=item_features_test,
            num_threads=args.threads, k=k
        )
        p = eval_res["p"]
        r = eval_res["r"]
        f1 = eval_res["f1"]
        print(f"k={k}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")
        evaluation.extend([
            {"k": k, "evaluation": "precisions", "value": p},
            {"k": k, "evaluation": "recall", "value": r},
            {"k": k, "evaluation": "f1", "value": f1}
        ])

    df_eval = pd.DataFrame(evaluation)
    eval_csv = f"result_evaluation/fm_feature({feature_name})_ml-{ml_size}_UMR({min_user_rating}).csv"
    df_eval.to_csv(eval_csv, index=False)

    plt.figure()
    plt.ylim(0, 1.0)
    ax = sns.lineplot(data=df_eval, x="k", y="value", style="evaluation", hue="evaluation", markers=True, dashes=False)
    ax.set(title=f'LightFM Feature ({feature_name}) - ml-{ml_size}', xlabel='TOP_K', ylabel='Score')
    fig = ax.get_figure()
    name_fig = f"evaluation/fm_feature({feature_name})_ml-{ml_size}_UMR({min_user_rating}).png"
    fig.savefig(name_fig)
    plt.close()
    print(f"Done. Evaluation saved to {eval_csv} and {name_fig}")


if __name__ == "__main__":
    main()
