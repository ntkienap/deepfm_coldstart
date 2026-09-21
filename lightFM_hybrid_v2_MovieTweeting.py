import os
import random
import pickle

import seaborn as sns
import numpy as np
import pandas as pd

import support_metric as spMetric
import support_dataset as spDataset
import matplotlib.pyplot as plt

from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import auc_score
from pathlib import Path
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k, auc, logloss

SIZE_VECTOR = 512

FEATURE_NAME = "SIFT" if SIZE_VECTOR == 64 else "VGG16"

PATH_FOLDER_DATASET = "dataset/MovieTweeting"
PATH_FILE_RATINGS = "MovieTweeting/ratings.dat"
PATH_FILE_LIGTHFM_MODEL = f"model/lightfm_with_feature({FEATURE_NAME})_MovieTweeting.model"

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

    item_matching_test = spDataset.get_image_features_MovieTweeting(
        ratings, isTest=True, size_vector=SIZE_VECTOR)

    dataset = Dataset()
    dataset.fit(ratings['userID'].unique(),
                ratings['itemID'].unique(),
                item_features=[f"f{i+1}" for i in range(SIZE_VECTOR)])

    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))

    item_features_test = dataset.build_item_features(((x[0], x[1])
                                                      for x in item_matching_test))
    ThuNghiem = True
    if (os.path.exists(PATH_FILE_LIGTHFM_MODEL) and not ThuNghiem):
        model = pickle.load(open(PATH_FILE_LIGTHFM_MODEL, 'rb'))
    else:
        (interactions_train, weights_train) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                         for _, x in ratings_train.iterrows())
        model = LightFM(loss='warp', item_alpha=0.00001)
        model_best = None
        auc_old = 0
        count = 0
        for epoch in range(150):
            item_matching_train = spDataset.get_image_features_MovieTweeting(ratings_train, size_vector=SIZE_VECTOR)
            item_features_train = dataset.build_item_features(((x[0], x[1])
                                                               for x in item_matching_train), normalize=False)
            model.fit_partial(interactions_train, sample_weight=weights_train,
                              item_features=item_features_train, epochs=2, verbose=True, num_threads=32)
            auc = spMetric.full_auc(model, interactions_train,
                            feature=item_features_test)
            print("auc_current;", auc, "; auc_before:", auc_old, "count:", count)
            if (auc - auc_old < 1e-4):
                count += 1
            elif (auc - auc_old > 1e-4):
                count = 0
            auc_old = auc
            if (count > 5):
                break
        pickle.dump(model, open(PATH_FILE_LIGTHFM_MODEL, 'wb'))

    print("Predict all user-item")
    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()
    all_predictions = spMetric.prepare_all_predictions(ratings, user_id_mapping, item_id_mapping,
                                                 interactions=interactions_train,
                                                 model=model, item_features=item_features_test,
                                                 num_threads=32)
    all_predictions.to_csv(
        f"prediction_all/MovieTweeting/fm_feature({FEATURE_NAME})_MovieTweeting.csv", index=False)
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
        f"result_evaluation/MovieTweeting/fm_feature({FEATURE_NAME})_MovieTweeting.csv", index=False)
    plt.ylim(0, 1.0)
    ax = sns.lineplot(
        data=df_eval,
        x="k",  y="value", style="evaluation", hue="evaluation", markers=True, dashes=False
    )
    ax.set(title='ĐÁNH GIÁ MÔ HÌNH FM WITH FEATURE')
    ax.set(xlabel='TOP_K', ylabel='Persent')
    fig = ax.get_figure()
    name_fig = f"evaluation/MovieTweeting/fm_feature({FEATURE_NAME})_MovieTweeting.png"
    print(name_fig)
    fig.savefig(name_fig)
