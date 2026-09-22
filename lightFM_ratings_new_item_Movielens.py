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
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")
DEFAULT_MIN_USER_RATING = int(os.environ.get("MIN_USER_RATING", "5"))


def main():
    parser = argparse.ArgumentParser(description="LightFM Ratings Only (New Item) on MovieLens")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--min-user-rating", type=int, default=DEFAULT_MIN_USER_RATING, help="Min user ratings threshold (default: 5)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 8), help="LightFM worker threads")
    parser.add_argument("--plot-hist", action="store_true", help="Generate histogram plots for ratings")
    args = parser.parse_args()

    ml_size = args.size
    min_user_rating = args.min_user_rating

    path_folder_dataset = "dataset"
    path_file_lightfm_model = f"model/lightfm_with_ratings_new_item_ml-{ml_size}_UMR({min_user_rating}).model"
    path_train_dataset = f"{path_folder_dataset}/train_new_item_ml-{ml_size}_UMR({min_user_rating}).pd"
    path_test_dataset = f"{path_folder_dataset}/test_new_item_ml-{ml_size}_UMR({min_user_rating}).pd"

    for d in [path_folder_dataset, "model", "prediction_all", "result_evaluation", "evaluation", "hist"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"Loading MovieLens [{ml_size}] with min_user_ratings={min_user_rating}...")
    ratings = spDataset.load_movielens_data(size=ml_size, min_user_ratings=min_user_rating)

    if os.path.exists(path_train_dataset) and os.path.exists(path_test_dataset):
        print(f"Loading cached train/test splits from {path_folder_dataset}...")
        ratings_train = pd.read_pickle(path_train_dataset)
        ratings_test = pd.read_pickle(path_test_dataset)
    else:
        print("Creating new item train/test split...")
        ratings_train, ratings_test = spDataset.get_train_test_dataset_new_items(ratings)
        ratings_train.to_pickle(path_train_dataset)
        ratings_test.to_pickle(path_test_dataset)

    print("Train shape:", ratings_train.shape, "Users:", ratings_train.userID.nunique(), "Items:", ratings_train.itemID.nunique())
    print("Test shape:", ratings_test.shape, "Users:", ratings_test.userID.nunique(), "Items:", ratings_test.itemID.nunique())

    if args.plot_hist:
        user_ratings_train = ratings_train.groupby(['userID'])['itemID'].count().reset_index().sort_values(["itemID"])
        item_ratings_train = ratings_train.groupby(['itemID'])['userID'].count().reset_index().sort_values(["userID"])
        item_ratings_test = ratings_test.groupby(['itemID'])['userID'].count().reset_index().sort_values(["userID"])
        user_ratings_test = ratings_test.groupby(['userID'])['itemID'].count().reset_index().sort_values(["itemID"])

        sns.barplot(data=item_ratings_train, x="itemID", y="userID")
        plt.ylabel("Số lượt đánh giá")
        plt.title("SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI ITEM")
        plt.savefig(f"hist/item_ratings_train_ml-{ml_size}_filter.png")
        plt.close()

        sns.barplot(data=user_ratings_train, x="userID", y="itemID")
        plt.ylabel("Số lượt đánh giá")
        plt.title("SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI USER")
        plt.savefig(f"hist/user_ratings_train_ml-{ml_size}_filter.png")
        plt.close()

        sns.barplot(data=item_ratings_test, x="itemID", y="userID")
        plt.ylabel("Số lượt đánh giá")
        plt.title("SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI ITEM")
        plt.savefig(f"hist/item_ratings_test_ml-{ml_size}_filter.png")
        plt.close()

        sns.barplot(data=user_ratings_test, x="userID", y="itemID")
        plt.ylabel("Số lượt đánh giá")
        plt.title("SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI USER")
        plt.savefig(f"hist/user_ratings_test_ml-{ml_size}_filter.png")
        plt.close()

    dataset = Dataset()
    dataset.fit(ratings['userID'].unique(), ratings['itemID'].unique())
    num_users, num_items = dataset.interactions_shape()
    print(f"Num users: {num_users}, num_items: {num_items}.")

    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    interactions_train, _ = dataset.build_interactions(
        (x['userID'], x['itemID'], x['rating']) for _, x in ratings_train.iterrows()
    )

    if os.path.exists(path_file_lightfm_model):
        print(f"Loading existing LightFM model from {path_file_lightfm_model}...")
        with open(path_file_lightfm_model, 'rb') as f:
            model = pickle.load(f)
    else:
        print(f"Training LightFM model ({args.epochs} epochs, {args.threads} threads)...")
        model = LightFM(no_components=41, loss='warp', learning_rate=0.001,
                        item_alpha=0.00022, user_alpha=0.00033, max_sampled=15)
        model.fit(interactions_train, epochs=args.epochs, verbose=True, num_threads=args.threads)
        with open(path_file_lightfm_model, 'wb') as f:
            pickle.dump(model, f)

    print("Predicting for all user-item pairs...")
    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()
    df_all_predictions = sp.prepare_all_predictions(
        ratings, user_id_mapping, item_id_mapping,
        interactions=interactions_train,
        model=model,
        num_threads=args.threads
    )
    df_all_predictions.to_csv(f"prediction_all/fm_ratings_new_item_ml-{ml_size}_UMR({min_user_rating}).csv", index=False)

    print("Evaluating Precision@k, Recall@k, F1@k...")
    evaluation = []
    for k in range(1, 21):
        p = precision_at_k(rating_true=test_df, rating_pred=df_all_predictions, k=k)
        r = recall_at_k(rating_true=test_df, rating_pred=df_all_predictions, k=k)
        f1 = 0 if (p + r == 0) else (2 * p * r) / (p + r)
        print(f"Precision@{k}: {p:.4f}, Recall@{k}: {r:.4f}, F1@{k}: {f1:.4f}")
        evaluation.extend([
            {"k": k, "evaluation": "precisions", "value": p},
            {"k": k, "evaluation": "recall", "value": r},
            {"k": k, "evaluation": "f1", "value": f1}
        ])

    df_eval = pd.DataFrame(evaluation)
    df_eval.to_csv(f"result_evaluation/fm_ratings_new_item_ml-{ml_size}_UMR({min_user_rating}).csv", index=False)

    plt.figure()
    plt.ylim(0, 1)
    ax = sns.lineplot(data=df_eval, x="k", y="value", style="evaluation", hue="evaluation", markers=True, dashes=False)
    ax.set(title=f'LightFM Ratings (New Item) - ml-{ml_size}', xlabel='TOP_K', ylabel='Score')
    fig = ax.get_figure()
    name_fig = f"evaluation/fm_ratings_new_items_ml-{ml_size}_UMR({min_user_rating}).png"
    fig.savefig(name_fig)
    plt.close()
    print(f"Done. Evaluation saved to {name_fig}")


if __name__ == "__main__":
    main()
