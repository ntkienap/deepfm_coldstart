import os
import random
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
import support_metric as sp
import matplotlib.pyplot as plt
import seaborn as sns
import support_dataset as spDataset

MOVIELENS_SIZE = os.environ.get("MOVIELENS_SIZE", "25m")
MIN_USER_RATING = int(os.environ.get("MIN_USER_RATING", "5"))

PATH_FOLDER_DATASET = "dataset"
PATH_FILE_LIGHTFM_MODEL = f"model/lightfm_with_ratings_new_item_ml_{MOVIELENS_SIZE}.model"
PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_new_item_ml_{MOVIELENS_SIZE}.pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_new_item_ml_{MOVIELENS_SIZE}.pd"

for d in [PATH_FOLDER_DATASET, "model", "evaluation"]:
    Path(d).mkdir(parents=True, exist_ok=True)


def predict_user_item(model, user_ids, item_ids):
    return model.predict(user_ids, item_ids=[item_ids])[0]


if __name__ == "__main__":
    ratings = spDataset.load_movielens_data(size=MOVIELENS_SIZE, min_user_ratings=MIN_USER_RATING)
    if os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset_new_items(ratings)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)

    dataset = Dataset()
    dataset.fit(ratings_train['userID'].unique(), ratings_train['itemID'].unique())
    new_iid = ratings_train['itemID'].max() + 1
    dataset.fit_partial(items=[new_iid])
    num_users, num_items = dataset.interactions_shape()
    print(f'Num users: {num_users}, num_items: {num_items}.')

    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    (interactions_train, weights_train) = dataset.build_interactions(
        (x['userID'], x['itemID'], x['rating']) for _, x in ratings_train.iterrows()
    )
    weights_train = weights_train.tocsr()
    train = interactions_train.tocsr()
    for uids, iids in zip(interactions_train.row, interactions_train.col):
        if weights_train[uids, iids] < 4:
            train[uids, iids] = 0
    interactions_train = train.tocoo()

    model = LightFM(loss='bpr', no_components=72, user_alpha=5e-3, item_alpha=5e-3, learning_rate=0.005)
    model.fit(interactions_train, sample_weight=weights_train.tocoo(), epochs=50, verbose=True, num_threads=8)
    pickle.dump(model, open(PATH_FILE_LIGHTFM_MODEL, 'wb'))

    print("Done test.")
