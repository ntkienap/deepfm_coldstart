import os
import pickle
import pandas as pd
import seaborn as sns
import support_metric as spMetric
from surprise import Dataset, Reader, SVD
from pathlib import Path
import support_dataset as spDataset
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k
import matplotlib.pyplot as plt

PATH_FOLDER_DATASET = "dataset/MovieTweeting"
FILE_MOVIE_MISSING_PATH = f"MovieTweeting/movie_id_poster_missing_MovieTweeting.csv"
PATH_FILE_RATINGS = f"MovieTweeting/ratings.dat"
PATH_FILE_SVD_MODEL = f"model/mf_svd_ratings_new_item_MovieTweeting.model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_new_item_MovieTweeting.pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_new_item_MovieTweeting.pd"

MIN_USER_RATINGS = 50
MIN_MOVIE_RATING = 5

if __name__ == "__main__":

    ratings = spDataset.load_movietweeting_data(
        PATH_FILE_RATINGS, min_user_ratings=MIN_USER_RATINGS, min_movie_ratings=MIN_MOVIE_RATING, sep="::")

    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset_new_items(ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)
    
    reader = Reader(rating_scale=(1, 10))
    data_train = Dataset.load_from_df(
        ratings_train[['userID', 'itemID', 'rating']], reader)

    
    data_train = data_train.build_full_trainset()

    model = SVD()
    
    model.fit(data_train)
    pickle.dump(model, open(PATH_FILE_SVD_MODEL, 'wb'))
    print("Predict all user-item")    
    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()
    df_all_predictions = spMetric.prepare_all_predictions_for_mf(
        ratings, ratings_train, model)

    df_all_predictions.to_csv(
        f"prediction_all/MovieTweeting/mf_new_item_MovieTweeting.csv", index=False)
    print("Top K..........")
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
        f"result_evaluation/MovieTweeting/mf_new_item_MovieTweeting.csv", index=False)
    plt.ylim(0, 1)

    ax = sns.lineplot(
        data=df_eval,
        x="k",  y="value", style="evaluation", hue="evaluation", markers=True, dashes=False
    )
    ax.set(title='ĐÁNH GIÁ MÔ HÌNH MF')
    ax.set(xlabel='TOP_K', ylabel='Persent')
    fig = ax.get_figure()
    name_fig = f"evaluation/MF_evaluation_ratings_new_item_MovieTweeting.png"
    print(name_fig)
    fig.savefig(name_fig)