import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import support_metric as spMetric

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")
DEFAULT_MIN_USER_RATING = int(os.environ.get("MIN_USER_RATING", "5"))
DEFAULT_FEATURE_NAME = os.environ.get("FEATURE_NAME", "VGG19")


def main():
    parser = argparse.ArgumentParser(description="Draw evaluation charts for MovieLens new item models")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--min-user-rating", type=int, default=DEFAULT_MIN_USER_RATING, help="Min user ratings threshold (default: 5)")
    parser.add_argument("--feature", default=DEFAULT_FEATURE_NAME, help="Feature name (default: VGG19)")
    args = parser.parse_args()

    ml_size = args.size
    min_user_rating = args.min_user_rating
    feature_name = args.feature

    path_folder_dataset = "dataset"
    path_test_dataset = f"{path_folder_dataset}/test_new_item_ml-{ml_size}_UMR({min_user_rating}).pd"
    Path("evaluation").mkdir(parents=True, exist_ok=True)
    Path("result_evaluation").mkdir(parents=True, exist_ok=True)

    if not os.path.exists(path_test_dataset):
        print(f"Error: {path_test_dataset} not found. Please train models or generate dataset first.")
        return

    ratings_test = pd.read_pickle(path_test_dataset)
    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()

    pred_feat_path = f"prediction_all/fm_feature({feature_name})_new_item_ml-{ml_size}_UMR({min_user_rating}).csv"
    pred_rate_path = f"prediction_all/fm_ratings_new_item_ml-{ml_size}_UMR({min_user_rating}).csv"
    pred_mf_path = f"prediction_all/mf_new_item_ml-{ml_size}_UMR({min_user_rating}).csv"

    if not (os.path.exists(pred_feat_path) and os.path.exists(pred_rate_path) and os.path.exists(pred_mf_path)):
        print(f"Warning: One or more prediction files not found:\n{pred_feat_path}\n{pred_rate_path}\n{pred_mf_path}")
        return

    df_feature_predictions_all = pd.read_csv(pred_feat_path, sep=",")
    df_ratings_predictions_all = pd.read_csv(pred_rate_path, sep=",")
    df_mf_predictions_all = pd.read_csv(pred_mf_path, sep=",")

    evaluation = []
    for k in range(1, 21):
        print(f"Evaluating Top_k@{k}...")
        ev = spMetric.precision_recall_all_model(
            test_df, df_feature_predictions_all, df_ratings_predictions_all, df_mf_predictions_all,
            k=k, feature_name=feature_name
        )
        evaluation.extend(ev)

    df = pd.DataFrame(evaluation)
    out_csv = f"result_evaluation/all_model_feature({feature_name})_new_item_ml-{ml_size}_UMR({min_user_rating}).csv"
    df.to_csv(out_csv, index=False)

    plt.figure()
    plt.ylim(0, 1)
    ax = sns.lineplot(data=df, x="recall", y="precisions", hue="model", style="model", markers=True, dashes=False)
    ax.set(title=f'PRECISION - RECALL (New Items, ml-{ml_size})')
    fig = ax.get_figure()
    fig.savefig(f"evaluation/precision_recall_feature({feature_name})_new_item_ml-{ml_size}_UMR({min_user_rating}).png")
    plt.close()

    plt.figure()
    plt.ylim(0, 1)
    ax = sns.lineplot(data=df, x="k", y="precisions", hue="model", style="model", markers=True, dashes=False)
    ax.set(title=f'PRECISION@K (New Items, ml-{ml_size})')
    fig = ax.get_figure()
    fig.savefig(f"evaluation/precision_feature({feature_name})_new_item_ml-{ml_size}_UMR({min_user_rating}).png")
    plt.close()

    plt.figure()
    plt.ylim(0, 1)
    ax = sns.lineplot(data=df, x="k", y="recall", hue="model", style="model", markers=True, dashes=True)
    ax.set(title=f'RECALL@K (New Items, ml-{ml_size})')
    fig = ax.get_figure()
    fig.savefig(f"evaluation/recall_feature({feature_name})_new_item_ml-{ml_size}_UMR({min_user_rating}).png")
    plt.close()

    plt.figure()
    plt.ylim(0, 1)
    ax = sns.lineplot(data=df, x="k", y="f1", hue="model", style="model", markers=True, dashes=True)
    ax.set(title=f'F1@K (New Items, ml-{ml_size})')
    fig = ax.get_figure()
    fig.savefig(f"evaluation/f1_feature({feature_name})_new_item_ml-{ml_size}_UMR({min_user_rating}).png")
    plt.close()

    print(f"Charts saved successfully for new items ml-{ml_size}.")


if __name__ == "__main__":
    main()
