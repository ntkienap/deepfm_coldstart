import copy
import os
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from lightfm.data import Dataset
from lightfm import LightFM
import support_metric as sp
import matplotlib.pyplot as plt
from lightfm.evaluation import auc_score


SIZE_VECTOR = 64
MOVILENS_SIZE = "1m"
PATH_ROOT_DATA = "/root/Data_LV_CuaDuong/"
PATH_FEATURE = f"{PATH_ROOT_DATA}/feature/BoW_{SIZE_VECTOR}"
EXTENSION_FEATURE = "bow"
FILE_MOVIE_MISSING_PATH = f"movielens/movie_id_poster_missing_1m.csv"

PATH_FOLDER_DATASET = "dataset/"
PATH_FILE_RATINGS = "movielens/ml-1m/ratings.dat"
PATH_FILE_LIGTHFM_MODEL_FEATURE = f"model/feature_{SIZE_VECTOR}_movielens_{MOVILENS_SIZE}.model"
PATH_FILE_LIGTHFM_MODEL_RATING = f"model/rating_movielens_{MOVILENS_SIZE}.model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_movielens_{MOVILENS_SIZE}.pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/train_movielens_{MOVILENS_SIZE}.pd"


def load_movielens_data(file_path='u.data', threshold=0, min_movie_ratings=5, min_user_ratings=5, sep="\t"):

    names = ['users', 'items', 'ratings', 'timestamp']
    dataframe = pd.read_csv(file_path, sep=sep, names=names, engine="python")
    movie_missing = pd.read_csv(FILE_MOVIE_MISSING_PATH, names=['items'])
    value_lists = movie_missing['items'].drop_duplicates(
    ).sort_values().values.tolist()

    dataframe = dataframe[~dataframe['items'].isin(value_lists)]
    dataframe = dataframe[dataframe['ratings'] >= threshold]

    filter_movies = (dataframe['items'].value_counts() > min_movie_ratings)
    filter_movies = filter_movies[filter_movies].index.tolist()

    filter_users = (dataframe['users'].value_counts() > min_user_ratings)
    filter_users = filter_users[filter_users].index.tolist()

    dataframe = dataframe[(dataframe['items'].isin(filter_movies)) & (
        dataframe['users'].isin(filter_users))]

    return dataframe


def get_image_features(df, isTest=False):
    items_set = df["items"].drop_duplicates(
    ).sort_values().reset_index()["items"]
    num_items = items_set.shape[0]
    item_matching = np.zeros((num_items * 1, 2), dtype=object)
    index = 0
    for idItem in items_set:
        if (isTest):
            pathFileFeature = f"{PATH_FEATURE}/{idItem}.bow"
        else:
            i = random.randint(0, 4)
            pathFileFeature = f"{PATH_FEATURE}/{idItem}_{i}.bow"
        if (os.path.exists(pathFileFeature)):
            with open(pathFileFeature) as in_feature:
                image_descriptor = np.loadtxt(in_feature, delimiter=",")
                item_matching[index] = [
                    idItem, {f"f{i+1}": v for i, v in enumerate(image_descriptor)}]
        else:
            item_matching[index] = [
                idItem, {f"f{i+1}": 0 for i in range(SIZE_VECTOR)}]
            #print(f"{idItem} error")
        index += 1
    return item_matching


def get_user_features(df):
    users = df["users"].drop_duplicates().sort_values().reset_index()["users"]
    dummies = pd.get_dummies(users, prefix="uid")
    user_features = np.zeros((users.shape[0], 2), dtype=object)
    for idx, row in dummies.iterrows():
        user_features[idx] = [users[idx], row.values.flatten().tolist()]
    return user_features


def get_train_test_dataset(df):
    df_train = df.groupby('users').sample(frac=0.75)
    df_test = df.drop(df_train.index)
    return df_train, df_test


def normalize(df, column):
    min = df[column].min()
    max = df[column].max()
    df[column] = (df[column] - min) / (max - min)


if __name__ == "__main__":

    ratings = load_movielens_data(PATH_FILE_RATINGS, sep="::")
    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = get_train_test_dataset(ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)
    dataset = Dataset()
    dataset.fit(ratings['users'].unique(),
                ratings['items'].unique(),
                item_features=[f"f{i+1}" for i in range(SIZE_VECTOR)])
    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))

    item_matching_train = get_image_features(ratings_train)
    item_matching_test = get_image_features(ratings_test, isTest=True)
    (interactions_train, weights_train) = dataset.build_interactions((x['users'], x['items'], x['ratings'])
                                                                     for _, x in ratings_train.iterrows())

    (interactions_test, weights_test) = dataset.build_interactions(((x['users'], x['items'], x['ratings'])
                                                                    for _, x in ratings_test.iterrows()))
    item_features_test = dataset.build_item_features(((x[0], x[1])
                                                     for x in item_matching_test), normalize=False)
    print(item_features_test.shape)

    model_feature = LightFM(loss='warp', no_components=64, user_alpha=1e-6, item_alpha=1e-6,
                            learning_rate=0.005)
    model_ratings = LightFM(loss='warp', no_components=64, user_alpha=1e-6, item_alpha=1e-6,
                            learning_rate=0.005)

    model_feature_best = None
    model_ratings_best = None
    auc_feature_best = 0
    auc_ratings_best = 0
    count = 0
    if (os.path.exists(PATH_FILE_LIGTHFM_MODEL_FEATURE) & os.path.exists(PATH_FILE_LIGTHFM_MODEL_RATING)):
        model_feature_best = pickle.load(
            open(PATH_FILE_LIGTHFM_MODEL_FEATURE, 'rb'))
        model_ratings_best = pickle.load(
            open(PATH_FILE_LIGTHFM_MODEL_RATING, 'rb'))
    else:
        (interactions_train, weights_train) = dataset.build_interactions((x['users'], x['items'], x['ratings'])
                                                                         for _, x in ratings_train.iterrows())
        model_ratings.fit(
            interactions_train, epochs=100, sample_weight=weights_train, num_threads=32)
        for epoch in range(100):
            print(f"epoch: {epoch+1}...............")
            item_matching = get_image_features(ratings_train)
            item_features_train = dataset.build_item_features(((x[0], x[1])
                                                               for x in item_matching), normalize=False)
            model_feature.fit_partial(
                interactions_train, epochs=3, item_features=item_features_train, sample_weight=weights_train, num_threads=32)
            auc = auc_score(model_feature, interactions_train,
                            item_features=item_features_train).mean()
            if (auc > auc_feature_best):
                model_feature_best = copy.deepcopy(model_feature)
                auc_feature_best = auc
                count = 0
            else:
                model_feature = copy.deepcopy(model_feature_best)
                count += 1
            if (count > 5):
                break

        pickle.dump(model_feature_best, open(
            PATH_FILE_LIGTHFM_MODEL_FEATURE, 'wb'))
        pickle.dump(model_ratings_best, open(
            PATH_FILE_LIGTHFM_MODEL_RATING, 'wb'))

    model_feature = model_feature_best
    model_ratings = model_ratings_best
    y_rank_feature = model_feature.predict_rank(
        test_interactions=interactions_test,
        item_features=item_features_test,
        num_threads=32
    )
    y_rank_ratings = model_ratings.predict_rank(
        test_interactions=interactions_test,
        num_threads=32
    )
    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    userIds = {y: x for x, y in user_id_mapping.items()}
    itemIds = {y: x for x, y in item_id_mapping.items()}

    dict_pred_feature = [{"UserId": userIds[uids],
                          "MovieId": itemIds[iids],
                          "Rating": ratings_test[(ratings_test["users"] == userIds[uids]) & (ratings_test["items"] == itemIds[iids])]["ratings"].values[0],
                          "Prediction": num_items - y_rank_feature[uids, iids],
                          "detail":""}
                         for uids, iids in zip(interactions_test.row, interactions_test.col)]

    dict_pred_ratings = [{"UserId": userIds[uids],
                          "MovieId": itemIds[iids],
                          "Rating": ratings_test[(ratings_test["users"] == userIds[uids]) & (ratings_test["items"] == itemIds[iids])]["ratings"].values[0],
                          "Prediction": num_items - y_rank_ratings[uids, iids],
                          "detail":""}
                         for uids, iids in zip(interactions_test.row, interactions_test.col)]
    df_pred_feature = pd.DataFrame(dict_pred_feature)
    df_pred_ratings = pd.DataFrame(dict_pred_ratings)
    normalize(df_pred_feature, "Prediction")
    normalize(df_pred_feature, "Rating")
    normalize(df_pred_ratings, "Prediction")
    normalize(df_pred_ratings, "Rating")
    avg = df_pred_feature["Rating"].mean()
    k_feature_precision = []
    k_feature_recall = []
    k_feature_f1 = []
    k_rating_precision = []
    k_rating_recall = []
    k_rating_f1 = []
    x = range(20, 60)
    for k in x:
        print(f"Top_k@{k}...............")
        precisions_feature, recalls_feature = sp.precision_recall_at_k(
            df_pred_feature.values, k=k, threshold=avg)
        precisions_ratings, recalls_ratings = sp.precision_recall_at_k(
            df_pred_ratings.values, k=k, threshold=avg)
        p = sum(prec for prec in precisions_feature.values()) / \
            len(precisions_feature)
        r = sum(rec for rec in recalls_feature.values()) / len(recalls_feature)
        f1 = (2 * p * r) / (p + r)
        k_feature_precision.append(p)
        k_feature_recall.append(r)
        k_feature_f1.append(f1)

        p = sum(prec for prec in precisions_ratings.values()) / \
            len(precisions_ratings)
        r = sum(rec for rec in recalls_ratings.values()) / len(recalls_ratings)
        f1 = (2 * p * r) / (p + r)
        k_rating_precision.append(p)
        k_rating_recall.append(r)
        k_rating_f1.append(f1)

    plt.title("PRECISION")
    plt.plot(x, np.array(k_feature_precision))
    plt.plot(x, np.array(k_rating_precision))
    plt.legend(['feature', 'ratings'], loc='lower right')
    plt.savefig(f"hist/precision_movielens_{MOVILENS_SIZE}.png")
    plt.close()

    plt.title("RECALL")
    plt.plot(x, np.array(k_feature_recall))
    plt.plot(x, np.array(k_rating_recall))
    plt.legend(['feature', 'ratings'], loc='lower right')
    plt.savefig(f"hist/recall_movielens_{MOVILENS_SIZE}.png")
    plt.close()

    plt.title("F1")
    plt.plot(x, np.array(k_feature_f1))
    plt.plot(x, np.array(k_rating_f1))
    plt.legend(['feature', 'ratings'], loc='lower right')
    plt.savefig(f"hist/f1_movielens_{MOVILENS_SIZE}.png")
    plt.close()
