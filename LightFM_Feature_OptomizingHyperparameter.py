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
from scipy.sparse import coo_matrix

SIZE_VECTOR = 512
MOVIELENS_SIZE = "100k"

PATH_FOLDER_DATASET = "dataset"
PATH_FILE_RATINGS = f"movielens/ml-{MOVIELENS_SIZE}/ratings.dat"
PATH_FILE_LIGTHFM_MODEL = f"model/ligthfm_with_feature_movielens_{MOVIELENS_SIZE}_VGG16.model"

# PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_ml-{MOVIELENS_SIZE}.pd"
# PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_ml-{MOVIELENS_SIZE}.pd"


PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_new_item_ml_{MOVIELENS_SIZE}_min_rating_50.pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_new_item_ml_{MOVIELENS_SIZE}_min_rating_50.pd"


def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    while True:
        yield {
           "no_components": np.random.randint(8, 32),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp"]),
            "learning_rate": round(np.random.uniform(1e-3, 1e-9),9),
            "item_alpha": round(np.random.uniform(1e-3, 1e-9),9),
            "user_alpha": round(np.random.uniform(1e-3, 1e-9),9),
            "max_sampled": np.random.randint(5, 25),
            "num_epochs": 10, #np.random.randint(5, 100),
        }


def random_search(ratings_train, train, test, feature_test, num_samples=50, num_threads=1):
    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        try:
            
            num_epochs = hyperparams.pop("num_epochs")
            model = LightFM(**hyperparams)
            for i in range(num_epochs):
                item_matching_train = spDataset.get_image_features(ratings_train)
                item_features_train = dataset.build_item_features(((x[0], x[1])
                                                                for x in item_matching_train), normalize=False)
                
                model.fit(train, epochs=1,
                      item_features=item_features_train, num_threads=num_threads, verbose=True)

            score = sp.full_auc(model, test, feature=feature_test)

            hyperparams["num_epochs"] = num_epochs
            print("Score {} at {}".format(score, hyperparams))
            with open(f"hyperparams/ligthfm_full_feature_ml-{MOVIELENS_SIZE}_new_item(0).txt", "a") as out_hyper:
                out_hyper.write("Score {} at {}\n".format(score, hyperparams))
            yield (score, hyperparams, model)
        except AttributeError:
            None


if __name__ == "__main__":
    ratings = spDataset.load_movielens_data(PATH_FILE_RATINGS, sep="::",min_user_ratings=50)

    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset(ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)

    dataset = Dataset()
    dataset.fit(ratings['userID'].unique(),
                ratings['itemID'].unique(),
                item_features=[f"f{i+1}" for i in range(SIZE_VECTOR)])

    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))

    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()

    item_matching_test = spDataset.get_image_features(
        ratings, isTest=True)
    item_features_test = dataset.build_item_features(((x[0], x[1])
                                                      for x in item_matching_test), normalize=False)

    (interactions_train, weights_train) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                     for _, x in ratings_train.iterrows())

    (interactions_test, weights_test) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                   for _, x in ratings_test.iterrows())

    
    with open(f"hyperparams/ligthfm_full_feature_ml-{MOVIELENS_SIZE}_new_item(0).txt", "w") as out_hyper:
        out_hyper.write("Full feature\n")
        
    (score, hyperparams, model) = max(random_search(ratings_train,
                                                    interactions_train, interactions_test, item_features_test, num_threads=32), key=lambda x: x[0])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
        
    print("Best score {} at {}".format(score, hyperparams))
    with open(f"hyperparams/ligthfm_full_feature_ml-{MOVIELENS_SIZE}_new_item(0).txt", "a") as out_hyper:
        out_hyper.write("** Best score {} at {}\n".format(score, hyperparams))
