import os
import sys
import pickle
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import support_metric as spm
import support_dataset as spDataset
from surprise import Dataset, Reader, SVD
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")
DEFAULT_MIN_USER_RATING = int(os.environ.get("MIN_USER_RATING", "5"))


def main():
    parser = argparse.ArgumentParser(description="Matrix Factorization (SVD) on MovieLens (New Items / Cold Start)")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--min-user-rating", type=int, default=DEFAULT_MIN_USER_RATING, help="Min user ratings threshold (default: 5)")
    args = parser.parse_args()

    ml_size = args.size
    min_user_rating = args.min_user_rating

    path_folder_dataset = "dataset"
    path_file_svd_model = f"model/mf_svd_new_item_ml-{ml_size}_UMR({min_user_rating}).model"
    path_train_dataset = f"{path_folder_dataset}/train_new_item_ml-{ml_size}_UMR({min_user_rating}).pd"
    path_test_dataset = f"{path_folder_dataset}/test_new_item_ml-{ml_size}_UMR({min_user_rating}).pd"

    for d in [path_folder_dataset, "model", "prediction_all", "result_evaluation", "evaluation"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"Loading MovieLens [{ml_size}] with min_user_ratings={min_user_rating} (Cold Start New Items)...")
    ratings = spDataset.load_movielens_data(size=ml_size, min_user_ratings=min_user_rating)

    if os.path.exists(path_train_dataset) and os.path.exists(path_test_dataset):
        print(f"Loading cached new item train/test splits from {path_folder_dataset}...")
        ratings_train = pd.read_pickle(path_train_dataset)
        ratings_test = pd.read_pickle(path_test_dataset)
    else:
        print("Creating new item cold start train/test split (90% train items / 10% new test items)...")
        ratings_train, ratings_test = spDataset.get_train_test_dataset_new_items(ratings)
        ratings_train.to_pickle(path_train_dataset)
        ratings_test.to_pickle(path_test_dataset)

    print(f"Train size: {len(ratings_train)} | Test size: {len(ratings_test)}")

    if os.path.exists(path_file_svd_model):
        print(f"Loading pre-trained SVD model from {path_file_svd_model}...")
        model = pickle.load(open(path_file_svd_model, 'rb'))
    else:
        print("Training SVD model...")
        reader = Reader(rating_scale=(1, 5))
        data_train = Dataset.load_from_df(
            ratings_train[['userID', 'itemID', 'rating']], reader)
        data_train = data_train.build_full_trainset()
        model = SVD()
        model.fit(data_train)
        pickle.dump(model, open(path_file_svd_model, 'wb'))
        print(f"Saved SVD model to {path_file_svd_model}")

    print("Generating predictions for cold start items...")
    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()
    df_all_predictions = spm.prepare_all_predictions_for_mf(
        ratings, ratings_train, model)

    pred_out_file = f"prediction_all/mf_new_item_ml-{ml_size}_UMR({min_user_rating}).csv"
    df_all_predictions.to_csv(pred_out_file, index=False)
    print(f"Saved predictions to {pred_out_file}")

    print("Evaluating Top-K metrics (Precision@K, Recall@K, F1@K)...")
    evaluation = []
    for k in range(1, 21):
        p = precision_at_k(rating_true=test_df, rating_pred=df_all_predictions, k=k)
        r = recall_at_k(test_df, df_all_predictions, k=k)
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        print(f"K={k:02d} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
        evaluation.extend([
            {"k": k, "evaluation": "precisions", "value": p},
            {"k": k, "evaluation": "recall", "value": r},
            {"k": k, "evaluation": "f1", "value": f1}
        ])

    df_eval = pd.DataFrame(evaluation)
    eval_csv_file = f"result_evaluation/mf_new_item_ml-{ml_size}_UMR({min_user_rating}).csv"
    df_eval.to_csv(eval_csv_file, index=False)

    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1)
    ax = sns.lineplot(
        data=df_eval,
        x="k", y="value", style="evaluation", hue="evaluation", markers=True, dashes=False
    )
    ax.set(title=f'ĐÁNH GIÁ MÔ HÌNH MF NEW ITEMS - MovieLens {ml_size.upper()}')
    ax.set(xlabel='Top-K', ylabel='Score')
    fig = ax.get_figure()
    plot_file = f"evaluation/mf_new_item_ml-{ml_size}_UMR({min_user_rating}).png"
    fig.savefig(plot_file)
    plt.close()
    print(f"Saved evaluation plot to {plot_file}")


if __name__ == "__main__":
    main()
