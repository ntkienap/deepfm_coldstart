import os
import pickle
import seaborn as sns
import random as rd
import pandas as pd
import numpy as np

import support_metric as sp
import support_dataset as spDataset
import matplotlib.pyplot as plt

from pathlib import Path
from lightfm.data import Dataset
from lightfm import LightFM
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k, auc, logloss

SIZE_VECTOR = 512
FEATURE_NAME = "SIFT" if SIZE_VECTOR == 64 else "VGG16"

PATH_FOLDER_DATASET = "dataset/MovieTweeting"
PATH_FILE_RATINGS = "MovieTweeting/ratings.dat"
PATH_FILE_LIGHTFM_MODEL = f"model/lightfm_with_feature({FEATURE_NAME})_new_item_MovieTweeting.model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_new_item.pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_new_item.pd"

MIN_USER_RATINGS = 50
MIN_MOVIE_RATING = 5


if __name__ == "__main__":
    ratings = spDataset.load_movietweeting_data(
        PATH_FILE_RATINGS, min_user_ratings=MIN_USER_RATINGS, min_movie_ratings=MIN_MOVIE_RATING, sep="::")

    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset_new_items(
            ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)

    dataset = Dataset()
    dataset.fit(ratings['userID'].unique(),
                ratings['itemID'].unique(),
                item_features=[f"f{i+1}" for i in range(SIZE_VECTOR)])

    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, Num items {}.'.format(num_users, num_items))

    (interactions_test, weights_test) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                   for _, x in ratings_test.iterrows())
    item_matching_test = spDataset.get_image_features_MovieTweeting(
        ratings, isTest=True, size_vector=SIZE_VECTOR)
    item_features = dataset.build_item_features(((x[0], x[1])
                                                 for x in item_matching_test), normalize=True)

    (interactions_train, weights_train) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                     for _, x in ratings_train.iterrows())
    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    ThuNghiem = False
    if (os.path.exists(PATH_FILE_LIGHTFM_MODEL) and not ThuNghiem):
        model = pickle.load(open(PATH_FILE_LIGHTFM_MODEL, 'rb'))
    else:
        model = LightFM(loss='warp', item_alpha=0.00001)
        auc_old = 0
        count = 0
        for epoch in range(250):
            item_matching_train = spDataset.get_image_features_MovieTweeting(
                ratings_train, size_vector=SIZE_VECTOR)
            item_features_train = dataset.build_item_features(((x[0], x[1])
                                                               for x in item_matching_train), normalize=True)
            model.fit_partial(interactions_train,
                              item_features=item_features_train, epochs=2, verbose=True, num_threads=32)

            auc = sp.full_auc(model, interactions_train, item_features)
            print("AUC Train Current:", auc,
                  "; AUC Train Old:", auc_old, "count:", count)
            if (auc - auc_old < 1e-4):
                count += 1
            elif (auc - auc_old > 1e-4):
                count = 0
            auc_old = auc
            if (count > 5):
                break

        pickle.dump(model, open(PATH_FILE_LIGHTFM_MODEL, 'wb'))

    print("Predict all user-item")
    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()
    all_predictions = sp.prepare_all_predictions(ratings, user_id_mapping, item_id_mapping,
                                                 interactions=interactions_train,
                                                 model=model, item_features=item_features,
                                                 num_threads=32)
    all_predictions.to_csv(
        f"prediction_all/MovieTweeting/fm_feature({FEATURE_NAME})_new_item_MovieTweeting.csv", index=False)
    print("Tinh K")
    evaluation = []

    x = range(1, 21)
    for k in x:
        p = precision_at_k(rating_true=test_df,
                           rating_pred=all_predictions, k=k)
        r = recall_at_k(test_df, all_predictions, k=k)
        f1 = (2 * p * r) / (p + r)
        print(f"Precision@{k}: {p}, Recall@{k}: {r}, F1@{k}: {f1}")
        evaluation.extend([{"k": k, "evaluation": "precisions", "value": p}, {
                          "k": k, "evaluation": "recall", "value": r}, {"k": k, "evaluation": "f1", "value": f1}])

    df_eval = pd.DataFrame(evaluation)
    df_eval.to_csv(
        f"result_evaluation/MovieTweeting/fm_feature({FEATURE_NAME})_new_item_MovieTweeting.csv", index=False)
    plt.ylim(0, 1.0)
    ax = sns.lineplot(
        data=df_eval,
        x="k",  y="value", style="evaluation", hue="evaluation", markers=True, dashes=False
    )
    ax.set(title=f"ĐÁNH GIÁ MÔ HÌNH {FEATURE_NAME}-FM")
    ax.set(xlabel='TOP_K', ylabel='Persent')
    fig = ax.get_figure()
    name_fig = f"evaluation/MovieTweeting/fm_feature({FEATURE_NAME})_new_item_MovieTweeting.png"
    print(name_fig)
    fig.savefig(name_fig)
