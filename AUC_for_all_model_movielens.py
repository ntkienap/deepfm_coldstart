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
import support_metric as spMetric

SIZE_VECTOR = 64
MOVIELENS_SIZE = "1m"
MIN_USER_RATING = 50
FEATURE_NAME = "SIFT" if SIZE_VECTOR == 64 else "VGG16"


PATH_FOLDER_DATASET = "dataset/"
PATH_FILE_RATINGS = f"movielens/ml-{MOVIELENS_SIZE}/ratings.dat"

PATH_FILE_LIGHTFM_MODEL_FEATURE_VGG16 = f"model/lightfm_with_feature(VGG16)_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).model"
PATH_FILE_LIGHTFM_MODEL_FEATURE_SIFT = f"model/lightfm_with_feature(SIFT)_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).model"
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
                item_features=[f"f{i+1}" for i in range(512)])
    dataset_sift = LightFMDataset()
    dataset_sift.fit(ratings['userID'].unique(),
                ratings['itemID'].unique(),
                item_features=[f"f{i+1}" for i in range(512)])


    (interactions_test, weights_test) = dataset_vgg16.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                   for _, x in ratings_test.iterrows())
    item_feature_matching = spDataset.get_image_features(
        ratings, isTest=True, size_vector=512)
    item_features_vgg16 = dataset_vgg16.build_item_features(((x[0], x[1])
                                                 for x in item_feature_matching), normalize=False)

    model_feature_vgg16 = pickle.load(
        open(PATH_FILE_LIGHTFM_MODEL_FEATURE_VGG16, 'rb'))
    
    model_ratings = pickle.load(
        open(PATH_FILE_LIGHTFM_MODEL_RATING, 'rb'))

    model_mf = pickle.load(
        open(PATH_FILE_SVD_MODEL, 'rb'))

    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()

    auc_ratings = spMetric.full_auc(model_ratings, interactions_test)
    auc_feature_vgg16 = spMetric.full_auc(
        model_feature_vgg16, interactions_test, feature=item_features_vgg16)
    auc_mf = spMetric.full_auc_mf(model_mf, interactions_test)
    df = pd.DataFrame(
        [{"model": f"VGG16-FM", "auc_score": auc_feature_vgg16},
         {"model": "FM", "auc_score": auc_ratings},
         {"model": "MF", "auc_score": auc_mf}])
    sns.barplot(data=df, x="model", y="auc_score")
    plt.savefig(f"evaluation/auc_score_feature(VGG16)_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png")
