import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
import support_metric as spMetric
import support_dataset as spDataset
import matplotlib.pyplot as plt
import seaborn as sns
from recommenders.evaluation.python_evaluation import auc, logloss, precision_at_k, recall_at_k

PATH_ROOT_DATA = "/root/Data_LV_CuaDuong/MovieTweeting"

PATH_FOLDER_DATASET = "dataset/MovieTweeting"
PATH_FILE_RATINGS = f"MovieTweeting/ratings.dat"
PATH_FILE_LIGTHFM_MODEL = f"model/ligthfm_with_ratings_MovieTweeting.model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train.pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test.pd"
MIN_USER_RATINGS = 50
MIN_MOVIE_RATING = 5

if __name__ == "__main__":
    ratings = spDataset.load_movietweeting_data(
        PATH_FILE_RATINGS, min_user_ratings=MIN_USER_RATINGS, min_movie_ratings=MIN_MOVIE_RATING, sep="::")

    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset(ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)
    print(ratings.shape, ratings.userID.unique().shape, ratings.itemID.unique().shape)
    print(ratings_train.shape, ratings_train.userID.unique().shape, ratings_train.itemID.unique().shape)
    print(ratings_test.shape, ratings_test.userID.unique().shape, ratings_test.itemID.unique().shape)
    
    exit(0)
    dataset = Dataset()
    dataset.fit(ratings['userID'].unique(),
                ratings['itemID'].unique())

    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))
    (interactions_train, weights_train) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                         for _, x in ratings_train.iterrows())
    (interactions_test, weights_test) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                   for _, x in ratings_test.iterrows())
    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()

    if (os.path.exists(PATH_FILE_LIGTHFM_MODEL)):
        model = pickle.load(open(PATH_FILE_LIGTHFM_MODEL, 'rb'))
    else:
        model = LightFM(no_components=41, loss='warp', learning_rate=0.001,
                        item_alpha=0.00022, user_alpha=0.00033, max_sampled=15)
        model.fit(interactions_train,
                  epochs=100, verbose=True, num_threads=32)
        pickle.dump(model, open(PATH_FILE_LIGTHFM_MODEL, 'wb'))
    
    print(spMetric.full_auc(model, interactions_test))
    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()

    evaluation = []
    print("Predict all user-item")
    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()
    df_all_predictions = spMetric.prepare_all_predictions(ratings,  uid_map=user_id_mapping,
                                                          iid_map=item_id_mapping,
                                                          interactions=interactions_train,
                                                          model=model,
                                                          num_threads=32)
    df_all_predictions.to_csv(
        f"prediction_all/MovieTweeting/fm_ratings_MovieTweeting.csv", index=False)
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
        f"result_evaluation/MovieTweeting/fm_ratings_MovieTweeting.csv", index=False)
    plt.ylim(0, 1)
    ax = sns.lineplot(
        data=df_eval,
        x="k",  y="value", style="evaluation", hue="evaluation", markers=True, dashes=False
    )
    ax.set(title='ĐÁNH GIÁ MÔ HÌNH FM WITH RATINGS')
    ax.set(xlabel='TOP_K', ylabel='Persent')
    fig = ax.get_figure()
    name_fig = f"evaluation/MovieTweeting/fm_ratings_MovieTweeting.png"
    print(name_fig)
    fig.savefig(name_fig)
