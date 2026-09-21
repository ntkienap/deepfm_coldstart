import os
import pickle
import pandas as pd
import seaborn as sns
import support_metric as spm
import support_dataset as spDataset
from surprise import Dataset, Reader, SVD, NMF
from pathlib import Path
import matplotlib.pyplot as plt
from recommenders.evaluation.python_evaluation import auc, logloss, precision_at_k, recall_at_k
from recommenders.models.surprise.surprise_utils import predict, compute_ranking_predictions
MOVIELENS_SIZE = "100k"
PATH_FOLDER_DATASET = "dataset"
MIN_USER_RATING = 5

PATH_FILE_RATINGS = f"movielens/ml-{MOVIELENS_SIZE}/ratings.dat"

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
        
    print(ratings.shape)
    print(ratings_train.shape)
    user_ratings_train = ratings_train.groupby(['userID'])['itemID'].count().reset_index().sort_values(["itemID"])
    print(user_ratings_train["itemID"].mean())
    print(user_ratings_train["itemID"].min())
    print(user_ratings_train["itemID"].max())

    print(ratings_test.shape)
    user_ratings_test = ratings_test.groupby(['userID'])['itemID'].count().reset_index().sort_values(["itemID"])
    print(user_ratings_test["itemID"].mean())
    print(user_ratings_test["itemID"].min())
    print(user_ratings_test["itemID"].max())