import os
import random
import pickle
import numpy as np
import pandas as pd
import support_metric as sp
import matplotlib.pyplot as plt
import seaborn as sns
from lightfm.evaluation import auc_score

from pathlib import Path
import support_metric as spMetric
import support_dataset as spDataset
from sklearn.preprocessing import minmax_scale

from surprise import Dataset as MFDataset, Reader, SVD
from lightfm.data import Dataset as LigthFMDataset

SIZE_VECTOR = 512

FEATURE_NAME = "VGG16"
PATH_ROOT_DATA = "/root/Data_LV_CuaDuong/MovieTweeting"

PATH_FOLDER_DATASET = "dataset/MovieTweeting"
PATH_FILE_RATINGS = f"MovieTweeting/ratings.dat"
PATH_FILE_LIGTHFM_MODEL_RATINGS = f"model/ligthfm_with_ratings_MovieTweeting.model"
PATH_FILE_LIGTHFM_MODEL_FEATURE = f"model/ligthfm_with_feature({FEATURE_NAME})_MovieTweeting.model"
PATH_FILE_SVD_MODEL = f"model/mf_svd_ratings_MovieTweeting.model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train.pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test.pd"
MIN_USER_RATINGS = 50
MIN_MOVIE_RATING = 5

if __name__ == "__main__":
    ratings = spDataset.load_movietweeting_data(PATH_FILE_RATINGS, sep="::",min_movie_ratings=MIN_MOVIE_RATING,min_user_ratings=MIN_USER_RATINGS)
    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset(ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)

  
    dataset = LigthFMDataset()
    dataset.fit(ratings['userID'].unique(),
                ratings['itemID'].unique(),
                item_features=[f"f{i+1}" for i in range(SIZE_VECTOR)])

    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))
    
    item_feature_matching = spDataset.get_image_features_MovieTweeting(
        ratings, isTest=True, size_vector=SIZE_VECTOR)
    item_features = dataset.build_item_features(((x[0], x[1])
                                                 for x in item_feature_matching), normalize=False)
    
    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    (interactions_train, weights_train) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                     for _, x in ratings_train.iterrows())
    model_feature = pickle.load(open(PATH_FILE_LIGTHFM_MODEL_FEATURE, 'rb'))
    model_ratings = pickle.load(open(PATH_FILE_LIGTHFM_MODEL_RATINGS, 'rb'))
    model_mf = pickle.load(open(PATH_FILE_SVD_MODEL, 'rb'))

    print("Predict all user-item")
    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()
    
    df_feature_predictions_all = sp.prepare_all_predictions(ratings, uid_map=user_id_mapping, iid_map=item_id_mapping,
                                                        interactions=interactions_train,
                                                        model=model_feature, item_features=item_features,
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
            test_df, df_feature_predictions_all, df_ratings_predictions_all, df_mf_predictions_all, k=k)
        print(ev)
        evaluation.extend(ev)

    df = pd.DataFrame(evaluation)
    df.to_csv(
        f"result_evaluation/MovieTweeting/all_model_feature({FEATURE_NAME})_MovieTweeting.csv", index=False)
    plt.ylim(0, 0.4)
    ax = sns.lineplot(
        data=df,
        x="recall",  y="precisions", hue="model", style="model", markers=True, dashes=False
    )
    ax.set(title='ĐỘ ĐO PRECISION - RECALL')

    fig = ax.get_figure()
    fig.savefig(
        f"evaluation/MovieTweeting/precision_recall_feature({FEATURE_NAME})_MovieTweeting.png")
    plt.close()

    plt.ylim(0, 0.4)
    ax = sns.lineplot(
        data=df,
        x="k",  y="precisions", hue="model", style="model", markers=True, dashes=False
    )
    ax.set(title='ĐỘ ĐO PRECISION')
    fig = ax.get_figure()
    fig.savefig(
        f"evaluation/MovieTweeting/precision_feature({FEATURE_NAME})_MovieTweeting.png")
    plt.close()

    plt.ylim(0, 0.4)
    ax = sns.lineplot(
        data=df,
        x="k",  y="recall", hue="model", style="model", markers=True, dashes=True
    )
    ax.set(title='ĐỘ ĐO RECALL')
    fig = ax.get_figure()
    fig.savefig(
        f"evaluation/MovieTweeting/recall_feature({FEATURE_NAME})_MovieTweeting.png")
    plt.close()

    plt.ylim(0, 0.4)
    ax = sns.lineplot(
        data=df,
        x="k",  y="f1", hue="model", style="model", markers=True, dashes=True
    )
    ax.set(title='ĐỘ ĐO F1')
    fig = ax.get_figure()
    fig.savefig(
        f"evaluation/f1__feature({FEATURE_NAME})_MovieTweeting.png")
    plt.close()
