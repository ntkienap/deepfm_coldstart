import os
import random
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from lightfm import LightFM
import support_metric as sp
import matplotlib.pyplot as plt
import seaborn as sns
import support_dataset as spDataset
import itertools
from lightfm import LightFM
from lightfm.evaluation import auc_score
import support_metric as sp

SIZE_VECTOR = 512
MOVIELENS_SIZE = "100k"

PATH_FOLDER_DATASET = "dataset"
PATH_FILE_RATINGS = f"movielens/ml-{MOVIELENS_SIZE}/ratings.dat"
PATH_FILE_LIGTHFM_MODEL = f"model/ligthfm_with_feature_movielens_{MOVIELENS_SIZE}_VGG16.model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_new_item_ml_{MOVIELENS_SIZE}.pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_new_item_ml_{MOVIELENS_SIZE}.pd"


def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    while True:
        yield {
            "no_components": np.random.randint(16, 64),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-3),
            "user_alpha": np.random.exponential(1e-3),
            "max_sampled": np.random.randint(5, 25),
            "num_epochs": np.random.randint(50, 150),
        }


def random_search(train, test, num_samples=150, num_threads=8):
    """
    Sample random hyperparameters, fit a LightFM model, and evaluate it
    on the test set.

    Parameters
    ----------

    train: np.float32 coo_matrix of shape [n_users, n_items]
        Training data.
    test: np.float32 coo_matrix of shape [n_users, n_items]
        Test data.
    num_samples: int, optional
        Number of hyperparameter choices to evaluate.


    Returns
    -------

    generator of (auc_score, hyperparameter dict, fitted model)

    """

    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)
        model.fit(train, epochs=num_epochs, num_threads=num_threads)

        score = sp.precision_at_k(model, test, k=1)

        hyperparams["num_epochs"] = num_epochs
        print("Score {} at {}".format(score, hyperparams))
        with open(f"hyperparams/ligthfm_ratings_ml-{MOVIELENS_SIZE}_new_item(-1).txt", "a") as out_hyper:
            out_hyper.write("Score {} at {}\n".format(score, hyperparams))
        yield (score, hyperparams, model)


if __name__ == "__main__":
    ratings = spDataset.load_movielens_data(PATH_FILE_RATINGS, sep="::")

    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset(ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)

    dataset = Dataset()
    dataset.fit(ratings['users'].unique(),
                ratings['items'].unique())

    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))

    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()

    # if (os.path.exists(PATH_FILE_LIGTHFM_MODEL)):
    #     model = pickle.load(open(PATH_FILE_LIGTHFM_MODEL, 'rb'))
    # else:
    (interactions_train, weights_train) = dataset.build_interactions((x['users'], x['items'], x['ratings'])
                                                                     for _, x in ratings_train.iterrows())
    (interactions_test, weights_test) = dataset.build_interactions((x['users'], x['items'], x['ratings'])
                                                                   for _, x in ratings_test.iterrows())
    train = spDataset.build_positive_data(interactions_train, weights_train)
    test = spDataset.build_positive_data(interactions_test, weights_test)
    (score, hyperparams, model) = max(random_search(
        train, test, num_threads=32), key=lambda x: x[0])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Best score {} at {}".format(score, hyperparams))
    with open(f"hyperparams/ligthfm_ratings_ml-{MOVIELENS_SIZE}_new_item(-1).txt", "a") as out_hyper:
        out_hyper.write("** Best score {} at {}\n".format(score, hyperparams))