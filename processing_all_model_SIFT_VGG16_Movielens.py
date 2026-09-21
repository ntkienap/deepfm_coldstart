import copy
import os
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from lightfm.data import Dataset as LightFMDataset
from lightfm import LightFM
from surprise import Dataset as MFDataset, Reader, SVD

import support_metric as sp
import matplotlib.pyplot as plt
import support_dataset as spDataset

SIZE_VECTOR_VGG16 = 512
SIZE_VECTOR_SIFT = 64
MOVIELENS_SIZE = "1m"
MIN_USER_RATING = 50
FEATURE_NAME_VGG16 = "VGG16"
FEATURE_NAME_SIFT = "SIFT"
FEATURE_NAME = "VGG16_SIFT"
PATH_FOLDER_DATASET = "dataset/"
PATH_FILE_RATINGS = f"movielens/ml-{MOVIELENS_SIZE}/ratings.dat"

PATH_FILE_LIGHTFM_MODEL_FEATURE_VGG16 = f"model/lightfm_with_feature({FEATURE_NAME_VGG16})_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).model"
PATH_FILE_LIGHTFM_MODEL_FEATURE_SIFT = f"model/lightfm_with_feature({FEATURE_NAME_SIFT})_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).model"
PATH_FILE_LIGHTFM_MODEL_RATING = f"model/lightfm_with_ratings_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).model"
PATH_FILE_SVD_MODEL = f"model/mf_svd_ratings_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).pd"


if __name__ == "__main__":

    ratings = spDataset.load_movielens_data(
        PATH_FILE_RATINGS, sep="::", min_user_ratings=MIN_USER_RATING)
    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset(ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)

    dataset_vgg16 = LightFMDataset()
    dataset_vgg16.fit(ratings['userID'].unique(),
                      ratings['itemID'].unique(),
                      item_features=[f"f{i+1}" for i in range(SIZE_VECTOR_VGG16)])

    dataset_sift = LightFMDataset()
    dataset_sift.fit(ratings['userID'].unique(),
                     ratings['itemID'].unique(),
                     item_features=[f"f{i+1}" for i in range(SIZE_VECTOR_SIFT)])

    user_id_mapping, _, item_id_mapping, _ = dataset_vgg16.mapping()

    (interactions_train, weights_train) = dataset_vgg16.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                           for _, x in ratings_train.iterrows())
    item_feature_matching = spDataset.get_image_features(
        ratings, isTest=True, size_vector=SIZE_VECTOR_VGG16)
    item_features_vgg16 = dataset_vgg16.build_item_features(((x[0], x[1])
                                                             for x in item_feature_matching), normalize=True)

    item_feature_matching = spDataset.get_image_features(
        ratings, isTest=True, size_vector=SIZE_VECTOR_SIFT)
    item_features_sift = dataset_sift.build_item_features(((x[0], x[1])
                                                           for x in item_feature_matching), normalize=True)

    num_users, num_items = dataset_vgg16.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))
    model_feature_vgg16 = pickle.load(
        open(PATH_FILE_LIGHTFM_MODEL_FEATURE_VGG16, 'rb'))

    model_feature_sift = pickle.load(
        open(PATH_FILE_LIGHTFM_MODEL_FEATURE_SIFT, 'rb'))

    model_ratings = pickle.load(
        open(PATH_FILE_LIGHTFM_MODEL_RATING, 'rb'))

    model_mf = pickle.load(
        open(PATH_FILE_SVD_MODEL, 'rb'))

    print("Predict all user-item")
    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()

    df_feature_vgg16_predictions_all = sp.prepare_all_predictions(ratings, uid_map=user_id_mapping, iid_map=item_id_mapping,
                                                                  interactions=interactions_train,
                                                                  model=model_feature_vgg16, item_features=item_features_vgg16,
                                                                  num_threads=32)
    df_feature_sift_predictions_all = sp.prepare_all_predictions(ratings, uid_map=user_id_mapping, iid_map=item_id_mapping,
                                                                 interactions=interactions_train,
                                                                 model=model_feature_sift, item_features=item_features_sift,
                                                                 num_threads=32)

    df_ratings_predictions_all = sp.prepare_all_predictions(ratings,  uid_map=user_id_mapping,
                                                            iid_map=item_id_mapping,
                                                            interactions=interactions_train,
                                                            model=model_ratings,
                                                            num_threads=32)

    df_mf_predictions_all = sp.prepare_all_predictions_for_mf(
        ratings, ratings_train, model_mf)

    evaluation = []
    x = range(1, 21)
    for k in x:
        print(f"Top_k@{k}...............")
        ev = sp.precision_recall_all_model(
            test_df, df_feature_vgg16_predictions_all, df_feature_sift_predictions_all, df_ratings_predictions_all, df_mf_predictions_all, k=k)
        print(ev)
        evaluation.extend(ev)

    df = pd.DataFrame(evaluation)
    df.to_csv(
        f"result_evaluation/all_model_feature({FEATURE_NAME})_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).csv", index=False)
    plt.ylim(0, 1)
    ax = sns.lineplot(
        data=df,
        x="recall",  y="precisions", hue="model", style="model", markers=True, dashes=False
    )
    ax.set(title='ĐỘ ĐO PRECISION - RECALL')

    fig = ax.get_figure()
    fig.savefig(
        f"evaluation/precision_recall_feature({FEATURE_NAME})_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png")
    plt.close()

    plt.ylim(0, 1)
    ax = sns.lineplot(
        data=df,
        x="k",  y="precisions", hue="model", style="model", markers=True, dashes=False
    )
    ax.set(title='ĐỘ ĐO PRECISION')
    fig = ax.get_figure()
    fig.savefig(
        f"evaluation/precision_feature({FEATURE_NAME})_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png")
    plt.close()

    plt.ylim(0, 1)
    ax = sns.lineplot(
        data=df,
        x="k",  y="recall", hue="model", style="model", markers=True, dashes=True
    )
    ax.set(title='ĐỘ ĐO RECALL')
    fig = ax.get_figure()
    fig.savefig(
        f"evaluation/recall_feature({FEATURE_NAME})_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png")
    plt.close()

    plt.ylim(0, 1)
    ax = sns.lineplot(
        data=df,
        x="k",  y="f1", hue="model", style="model", markers=True, dashes=True
    )
    ax.set(title='ĐỘ ĐO F1')
    fig = ax.get_figure()
    fig.savefig(
        f"evaluation/f1_feature({FEATURE_NAME})_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png")
    plt.close()
