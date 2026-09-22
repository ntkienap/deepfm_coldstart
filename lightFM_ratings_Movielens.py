import os
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
import support_metric as sp
import support_dataset as spDataset
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")
DEFAULT_MIN_USER_RATING = int(os.environ.get("MIN_USER_RATING", "5"))


def main():
    parser = argparse.ArgumentParser(description="LightFM Ratings Only on MovieLens")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--min-user-rating", type=int, default=DEFAULT_MIN_USER_RATING, help="Min user ratings threshold (default: 5)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 8), help="LightFM worker threads")
    args = parser.parse_args()

    ml_size = args.size
    min_user_rating = args.min_user_rating

    path_folder_dataset = "dataset"
    path_file_lightfm_model = f"model/lightfm_with_ratings_ml-{ml_size}_UMR({min_user_rating}).model"
    path_train_dataset = f"{path_folder_dataset}/train_ml-{ml_size}_UMR({min_user_rating}).pd"
    path_test_dataset = f"{path_folder_dataset}/test_ml-{ml_size}_UMR({min_user_rating}).pd"

    for d in [path_folder_dataset, "model", "prediction_all", "result_evaluation", "evaluation"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    env = {
        "FEATURE_NAME": "NONE",
        "MOVIELENS_SIZE": ml_size,
        "MIN_USER_RATING": min_user_rating,
        "isNew": False
    }

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

    dataset = Dataset()
    dataset.fit(ratings['userID'].unique(), ratings['itemID'].unique())

    num_users, num_items = dataset.interactions_shape()
    print(f"Number of users: {num_users} | Number of items: {num_items}")

    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    interactions_train, weights_train = dataset.build_interactions(
        (x['userID'], x['itemID'], x['rating']) for _, x in ratings_train.iterrows()
    )

    if os.path.exists(path_file_lightfm_model):
        print(f"Loading pre-trained LightFM model from {path_file_lightfm_model}...")
        model = pickle.load(open(path_file_lightfm_model, 'rb'))
    else:
        print(f"Training LightFM model for {args.epochs} epochs with {args.threads} threads...")
        model = LightFM(no_components=41, loss='warp', learning_rate=0.001,
                        item_alpha=0.00022, user_alpha=0.00033, max_sampled=15)
        model.fit(interactions_train, epochs=args.epochs, verbose=True, num_threads=args.threads)
        pickle.dump(model, open(path_file_lightfm_model, 'wb'))
        print(f"Saved model to {path_file_lightfm_model}")

    print("Evaluating Top-K metrics (Precision@K, Recall@K, F1@K)...")
    evaluation = []
    for k in range(1, 21):
        df_test = ratings_test[['userID', 'itemID', 'rating']].copy()
        eval_k = sp.precision_recall_f1_at_k(
            ratings, user_id_mapping, item_id_mapping,
            df_test=df_test,
            model=model, env=env, item_features=None,
            num_threads=args.threads, k=k
        )
        print(f"K={k:02d} | Precision: {eval_k['p']:.4f} | Recall: {eval_k['r']:.4f} | F1: {eval_k['f1']:.4f}")
        evaluation.extend([
            {"k": k, "evaluation": "precisions", "value": eval_k["p"]},
            {"k": k, "evaluation": "recall", "value": eval_k["r"]},
            {"k": k, "evaluation": "f1", "value": eval_k["f1"]}
        ])

    df_eval = pd.DataFrame(evaluation)
    eval_csv_file = f"result_evaluation/fm_ratings_ml-{ml_size}_UMR({min_user_rating}).csv"
    df_eval.to_csv(eval_csv_file, index=False)

    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1)
    ax = sns.lineplot(
        data=df_eval,
        x="k", y="value", style="evaluation", hue="evaluation", markers=True, dashes=False
    )
    ax.set(title=f'ĐÁNH GIÁ MÔ HÌNH FM WITH RATINGS - MovieLens {ml_size.upper()}')
    ax.set(xlabel='Top-K', ylabel='Score')
    fig = ax.get_figure()
    plot_file = f"evaluation/fm_ratings_ml-{ml_size}_UMR({min_user_rating}).png"
    fig.savefig(plot_file)
    plt.close()
    print(f"Saved evaluation plot to {plot_file}")


if __name__ == "__main__":
    main()
