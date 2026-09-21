from cgi import test
import os
import pickle
import random
import pandas as pd
import seaborn as sns
import support_metric as spm
from surprise import Dataset, Reader, SVD, NMF
from pathlib import Path
import matplotlib.pyplot as plt
import support_dataset as spDataset
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k, auc

MOVIELENS_SIZE = "1m"
PATH_FOLDER_DATASET = "dataset"
MIN_USER_RATING = 50

PATH_FILE_RATINGS = f"movielens/ml-{MOVIELENS_SIZE}/ratings.dat"
PATH_FILE_SVD_MODEL = f"model/mf_svd_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).pd"

if __name__ == "__main__":
    ratings = spDataset.load_movielens_data(
        PATH_FILE_RATINGS, sep="::", min_user_ratings=MIN_USER_RATING)
    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset_new_items(
            ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)

    if (os.path.exists(PATH_FILE_SVD_MODEL)):
        model = pickle.load(open(PATH_FILE_SVD_MODEL, 'rb'))
    else:
        reader = Reader(rating_scale=(1, 5))
        data_train = Dataset.load_from_df(
        ratings_train[['userID', 'itemID', 'rating']], reader)
        data_train = data_train.build_full_trainset()
        model = SVD()
        model.fit(data_train)
        pickle.dump(model, open(PATH_FILE_SVD_MODEL, 'wb'))

    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()
    df_all_predictions = spm.prepare_all_predictions_for_mf(
        ratings, ratings_train, model)

    df_all_predictions.to_csv(
        f"prediction_all/mf_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).csv", index=False)
    evaluation = []
    x = range(1, 21)
    for k in x:
        p = precision_at_k(rating_true=test_df,
                           rating_pred=df_all_predictions, k=k)
        r = recall_at_k(test_df, df_all_predictions, k=k)
        f1 = (2 * p * r) / (p + r)
        print(f"Precision@{k}: {p}, Recall@{k}: {r}, F1@{k}: {f1}")
        evaluation.extend([{"k": k, "evaluation": "precisions", "value": p}, {
                          "k": k, "evaluation": "recall", "value": r}, {"k": k, "evaluation": "f1", "value": f1}])
    df_eval = pd.DataFrame(evaluation)
    df_eval.to_csv(
        f"result_evaluation/mf_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).csv", index=False)
    plt.ylim(0, 1)

    ax = sns.lineplot(
        data=df_eval,
        x="k",  y="value", style="evaluation", hue="evaluation", markers=True, dashes=False
    )
    ax.set(title='ĐÁNH GIÁ MÔ HÌNH MF WITH RATINGS')
    ax.set(xlabel='TOP_K', ylabel='Persent')
    fig = ax.get_figure()
    name_fig = f"evaluation/mf_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png"
    print(name_fig)
    fig.savefig(name_fig)
