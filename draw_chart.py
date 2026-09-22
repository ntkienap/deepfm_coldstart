import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DEFAULT_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")


def main():
    parser = argparse.ArgumentParser(description="Draw chart from Movielens summary CSV")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="MovieLens dataset size (default: 25m)")
    parser.add_argument("--csv-file", default=None, help="Path to CSV file (defaults to result_evaluation/Movielens_{size}.csv)")
    args = parser.parse_args()

    ml_size = args.size
    csv_file = args.csv_file or f"result_evaluation/Movielens_{ml_size}.csv"

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} does not exist.")
        return

    Path("evaluation").mkdir(parents=True, exist_ok=True)
    names = ["index", "k", "type", "precisions", "recall", "f1"]
    df = pd.read_csv(csv_file, header=0, names=names)
    if "index" in df.columns:
        df.drop(columns=["index"], inplace=True)

    plt.figure()
    ax = sns.lineplot(data=df, x="recall", y="precisions", hue="type", style="type", markers=True, dashes=False)
    ax.set(title=f'PRECISION - RECALL (ml-{ml_size})')
    plt.savefig(f"evaluation/precision_recall_Movielens_{ml_size}.png")
    plt.close()

    plt.figure()
    ax = sns.lineplot(data=df, x="k", y="precisions", hue="type", style="type", markers=True, dashes=False)
    ax.set(title=f'PRECISION (ml-{ml_size})')
    fig = ax.get_figure()
    fig.savefig(f"evaluation/precision_Movielens_{ml_size}.png")
    plt.close()

    plt.figure()
    ax = sns.lineplot(data=df, x="k", y="recall", hue="type", style="type", markers=True, dashes=True)
    ax.set(title=f'RECALL (ml-{ml_size})')
    fig = ax.get_figure()
    fig.savefig(f"evaluation/recall_Movielens_{ml_size}.png")
    plt.close()

    plt.figure()
    ax = sns.lineplot(data=df, x="k", y="f1", hue="type", style="type", markers=True, dashes=True)
    ax.set(title=f'F1 (ml-{ml_size})')
    fig = ax.get_figure()
    fig.savefig(f"evaluation/f1_Movielens_{ml_size}.png")
    plt.close()
    print(f"Plots saved to evaluation/ for ml-{ml_size}.")


if __name__ == "__main__":
    main()
