import os
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lightfm.data import Dataset as LightFMDataset
import support_metric as sp
import support_dataset as spDataset

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")
DEFAULT_MIN_USER_RATING = int(os.environ.get("MIN_USER_RATING", "5"))
DEFAULT_FEATURE_NAME = os.environ.get("FEATURE_NAME", "VGG19")
DEFAULT_POOLING = os.environ.get("POOLING", "avg")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate All Models on MovieLens (Standard)")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--min-user-rating", type=int, default=DEFAULT_MIN_USER_RATING, help="Min user ratings threshold (default: 5)")
    parser.add_argument("--feature", default=DEFAULT_FEATURE_NAME, choices=["VGG19", "ResNet50", "SIFT", "BoW"], help="Feature name (default: VGG19)")
    parser.add_argument("--size-vector", type=int, default=None, help="Vector dimension")
    parser.add_argument("--pooling", default=DEFAULT_POOLING, choices=["avg", "max"], help="Pooling (default: avg)")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 8), help="Worker threads")
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
    path_file_lightfm_model_feature = f"model/lightfm_with_feature({feature_name})_ml-{ml_size}_UMR({min_user_rating}).model"
    path_file_lightfm_model_rating = f"model/lightfm_with_ratings_ml-{ml_size}_UMR({min_user_rating}).model"
    path_file_svd_model = f"model/mf_svd_ratings_ml-{ml_size}_UMR({min_user_rating}).model"

    path_train_dataset = f"{path_folder_dataset}/train_ml-{ml_size}_UMR({min_user_rating}).pd"
    path_test_dataset = f"{path_folder_dataset}/test_ml-{ml_size}_UMR({min_user_rating}).pd"

    for d in [path_folder_dataset, "model", "prediction_all", "result_evaluation", "evaluation"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"Loading MovieLens [{ml_size}] with min_user_ratings={min_user_rating}...")
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

    dataset = LightFMDataset()
    dataset.fit(
        ratings['userID'].unique(),
        ratings['itemID'].unique(),
        item_features=[f"f{i+1}" for i in range(size_vector)]
    )
    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()

    (interactions_train, _) = dataset.build_interactions(
        (x['userID'], x['itemID'], x['rating']) for _, x in ratings_train.iterrows()
    )

    print("Building visual features for items...")
    item_feature_matching = spDataset.get_image_features(
        ratings, isTest=True, size_vector=size_vector, pooling=pooling, feature_name=feature_name
    )
    item_features = dataset.build_item_features(
        ((x[0], x[1]) for x in item_feature_matching), normalize=False
    )

    num_users, num_items = dataset.interactions_shape()
    print(f"Num users: {num_users}, Num items: {num_items}.")

    print("Loading models...")
    assert os.path.exists(path_file_lightfm_model_feature), f"Missing model: {path_file_lightfm_model_feature}"
    assert os.path.exists(path_file_lightfm_model_rating), f"Missing model: {path_file_lightfm_model_rating}"
    assert os.path.exists(path_file_svd_model), f"Missing model: {path_file_svd_model}"

    with open(path_file_lightfm_model_feature, 'rb') as f:
        model_feature = pickle.load(f)
    with open(path_file_lightfm_model_rating, 'rb') as f:
        model_ratings = pickle.load(f)
    with open(path_file_svd_model, 'rb') as f:
        model_mf = pickle.load(f)

    print("Generating predictions for all models...")
    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()

    df_feature_predictions_all = sp.prepare_all_predictions(
        ratings, uid_map=user_id_mapping, iid_map=item_id_mapping,
        interactions=interactions_train, model=model_feature,
        item_features=item_features, num_threads=args.threads
    )
    df_ratings_predictions_all = sp.prepare_all_predictions(
        ratings, uid_map=user_id_mapping, iid_map=item_id_mapping,
        interactions=interactions_train, model=model_ratings,
        num_threads=args.threads
    )
    df_mf_predictions_all = sp.prepare_all_predictions_for_mf(
        ratings, ratings_train, model_mf
    )

    print("Evaluating Top-K metrics (1 to 20)...")
    evaluation = []
    for k in range(1, 21):
        print(f"Top_k@{k}...")
        ev = sp.precision_recall_all_model(
            test_df, df_feature_predictions_all, df_ratings_predictions_all, df_mf_predictions_all, k=k
        )
        evaluation.extend(ev)

    df_eval = pd.DataFrame(evaluation)
    out_csv = f"result_evaluation/all_model_feature({feature_name})_ml-{ml_size}_UMR({min_user_rating}).csv"
    df_eval.to_csv(out_csv, index=False)

    plt.figure()
    plt.ylim(0, 1)
    ax = sns.lineplot(data=df_eval, x="recall", y="precisions", hue="model", style="model", markers=True, dashes=False)
    ax.set(title=f'PRECISION - RECALL (ml-{ml_size})')
    fig = ax.get_figure()
    fig.savefig(f"evaluation/precision_recall_feature({feature_name})_ml-{ml_size}_UMR({min_user_rating}).png")
    plt.close()

    plt.figure()
    plt.ylim(0, 1)
    ax = sns.lineplot(data=df_eval, x="k", y="precisions", hue="model", style="model", markers=True, dashes=False)
    ax.set(title=f'PRECISION@K (ml-{ml_size})')
    fig = ax.get_figure()
    fig.savefig(f"evaluation/precision_feature({feature_name})_ml-{ml_size}_UMR({min_user_rating}).png")
    plt.close()

    plt.figure()
    plt.ylim(0, 1)
    ax = sns.lineplot(data=df_eval, x="k", y="recall", hue="model", style="model", markers=True, dashes=True)
    ax.set(title=f'RECALL@K (ml-{ml_size})')
    fig = ax.get_figure()
    fig.savefig(f"evaluation/recall_feature({feature_name})_ml-{ml_size}_UMR({min_user_rating}).png")
    plt.close()

    plt.figure()
    plt.ylim(0, 1)
    ax = sns.lineplot(data=df_eval, x="k", y="f1", hue="model", style="model", markers=True, dashes=True)
    ax.set(title=f'F1@K (ml-{ml_size})')
    fig = ax.get_figure()
    fig.savefig(f"evaluation/f1_feature({feature_name})_ml-{ml_size}_UMR({min_user_rating}).png")
    plt.close()

    print(f"Done! Results saved to {out_csv} and evaluation plots.")


if __name__ == "__main__":
    main()
