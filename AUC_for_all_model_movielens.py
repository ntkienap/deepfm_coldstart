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
import support_dataset as spDataset
import support_metric as spMetric

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")
DEFAULT_MIN_USER_RATING = int(os.environ.get("MIN_USER_RATING", "5"))
DEFAULT_FEATURE_NAME = os.environ.get("FEATURE_NAME", "VGG19")
DEFAULT_POOLING = os.environ.get("POOLING", "avg")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute AUC for All Models on MovieLens")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--min-user-rating", type=int, default=DEFAULT_MIN_USER_RATING, help="Min user ratings threshold (default: 5)")
    parser.add_argument("--feature", default=DEFAULT_FEATURE_NAME, choices=["VGG19", "ResNet50", "SIFT", "BoW"], help="Feature name (default: VGG19)")
    parser.add_argument("--size-vector", type=int, default=None, help="Feature vector size")
    parser.add_argument("--pooling", default=DEFAULT_POOLING, choices=["avg", "max"], help="Pooling (default: avg)")
    return parser.parse_args()


def main():
    args = parse_args()
    ml_size = args.size
    min_user_rating = args.min_user_rating
    feature_name = args.feature

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
    path_file_model_feature = f"model/lightfm_with_feature({feature_name})_ml-{ml_size}_UMR({min_user_rating}).model"
    path_file_model_rating = f"model/lightfm_with_ratings_ml-{ml_size}_UMR({min_user_rating}).model"
    path_file_model_svd = f"model/mf_svd_ratings_ml-{ml_size}_UMR({min_user_rating}).model"

    path_train_dataset = f"{path_folder_dataset}/train_ml-{ml_size}_UMR({min_user_rating}).pd"
    path_test_dataset = f"{path_folder_dataset}/test_ml-{ml_size}_UMR({min_user_rating}).pd"

    Path("evaluation").mkdir(parents=True, exist_ok=True)

    print(f"Loading MovieLens [{ml_size}] with min_user_ratings={min_user_rating}...")
    ratings = spDataset.load_movielens_data(size=ml_size, min_user_ratings=min_user_rating)

    if os.path.exists(path_train_dataset) and os.path.exists(path_test_dataset):
        ratings_test = pd.read_pickle(path_test_dataset)
    else:
        _, ratings_test = spDataset.get_train_test_dataset(ratings)

    dataset = LightFMDataset()
    dataset.fit(
        ratings['userID'].unique(),
        ratings['itemID'].unique(),
        item_features=[f"f{i+1}" for i in range(size_vector)]
    )

    (interactions_test, _) = dataset.build_interactions(
        (x['userID'], x['itemID'], x['rating']) for _, x in ratings_test.iterrows()
    )

    item_feature_matching = spDataset.get_image_features(
        ratings, isTest=True, size_vector=size_vector, pooling=args.pooling, feature_name=feature_name
    )
    item_features = dataset.build_item_features(
        ((x[0], x[1]) for x in item_feature_matching), normalize=False
    )

    print("Loading models to evaluate AUC...")
    results = []

    if os.path.exists(path_file_model_feature):
        with open(path_file_model_feature, 'rb') as f:
            model_feature = pickle.load(f)
        auc_feat = spMetric.full_auc(model_feature, interactions_test, feature=item_features)
        results.append({"model": f"{feature_name}-FM", "auc_score": auc_feat})
        print(f"{feature_name}-FM AUC: {auc_feat:.4f}")

    if os.path.exists(path_file_model_rating):
        with open(path_file_model_rating, 'rb') as f:
            model_ratings = pickle.load(f)
        auc_ratings = spMetric.full_auc(model_ratings, interactions_test)
        results.append({"model": "FM", "auc_score": auc_ratings})
        print(f"FM AUC: {auc_ratings:.4f}")

    if os.path.exists(path_file_model_svd):
        with open(path_file_model_svd, 'rb') as f:
            model_mf = pickle.load(f)
        auc_mf = spMetric.full_auc_mf(model_mf, interactions_test)
        results.append({"model": "MF", "auc_score": auc_mf})
        print(f"MF AUC: {auc_mf:.4f}")

    if results:
        df = pd.DataFrame(results)
        plt.figure()
        sns.barplot(data=df, x="model", y="auc_score")
        plt.title(f"AUC Score Comparison (ml-{ml_size})")
        out_fig = f"evaluation/auc_score_feature({feature_name})_ml-{ml_size}_UMR({min_user_rating}).png"
        plt.savefig(out_fig)
        plt.close()
        print(f"Plot saved to {out_fig}")
    else:
        print("No trained models found to compute AUC.")


if __name__ == "__main__":
    main()
