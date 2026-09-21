import os
import pandas as pd
import random as rd
import numpy as np
import scipy.sparse as ssp

SIZE_VECTOR = 512

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = BASE_DIR if os.path.exists(os.path.join(BASE_DIR, "movielens")) else "/home/nckh/nckh_code"

def get_path_ext_feature_movielens(size_vector=SIZE_VECTOR, ext_feature = "VGG19"):
    PATH_ROOT_DATA = os.environ.get("PATH_ROOT_DATA", DEFAULT_ROOT)
    __path_feature = f"{PATH_ROOT_DATA}/feature/movielens/VGG19_avg_L1" if size_vector == 512 else f"{PATH_ROOT_DATA}/feature/movielens/BoW_64"
    __ext_feature = ext_feature if size_vector == 512 else "bow"
    return __path_feature, __ext_feature

def get_path_ext_feature_MovieTweeting(size_vector=SIZE_VECTOR, ext_feature = "VGG19"):
    PATH_ROOT_DATA = os.environ.get("PATH_ROOT_DATA", f"{DEFAULT_ROOT}/MovieTweeting")
    __path_feature = f"{PATH_ROOT_DATA}/feature/MovieTweeting/VGG19_max_L1" if size_vector == 512 else f"{PATH_ROOT_DATA}/feature/MovieTweeting/BoW_64"
    __ext_feature = ext_feature if size_vector == 512 else "bow"
    return __path_feature, __ext_feature

def load_movietweeting_data(file_path='MovieTweeting/ratings.dat', threshold=0, min_movie_ratings=1, min_user_ratings=1, sep="\t"):
    FILE_MOVIE_MISSING_PATH = f"MovieTweeting/movie_id_poster_missing_MovieTweeting.csv"
    names = ['userID', 'itemID', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep=sep, names=names,
                            engine="python")
    movie_missing = pd.read_csv(FILE_MOVIE_MISSING_PATH, names=['itemID'])
    value_lists = movie_missing['itemID'].drop_duplicates(
    ).sort_values().values.tolist()

    df = df[~df['itemID'].isin(value_lists)]
    df = df[df['rating'] >= threshold]
    while df['itemID'].value_counts().min() <= min_movie_ratings or df['userID'].value_counts().min() <= min_user_ratings:
        filter_movies = (df['itemID'].value_counts() > min_movie_ratings)
        filter_movies = filter_movies[filter_movies].index.tolist()

        filter_users = (df['userID'].value_counts() > min_user_ratings)
        filter_users = filter_users[filter_users].index.tolist()

        df = df[(df['itemID'].isin(filter_movies)) & (
            df['userID'].isin(filter_users))]

    return df

def get_image_features_MovieTweeting(df, isTest=False, size_vector=SIZE_VECTOR):
    __path_feature ,__ext_feature = get_path_ext_feature_MovieTweeting(size_vector)
    items_set = df["itemID"].drop_duplicates(
    ).sort_values().reset_index()["itemID"]
    num_items = items_set.shape[0]
    item_matching = np.zeros((num_items * 1, 2), dtype=object)
    index = 0
    i = None
    for idItem in items_set:
        if (isTest):
            pathFileFeature = f"{__path_feature}/{idItem}.{__ext_feature}"
        else:
            i = rd.randint(0, 4)
            pathFileFeature = f"{__path_feature}/{idItem}_{i}.{__ext_feature}"
        if (os.path.exists(pathFileFeature)):
            with open(pathFileFeature) as in_feature:
                image_descriptor = np.loadtxt(
                    in_feature, delimiter=",", dtype=np.float64)
                item_matching[index] = [
                    idItem, {f"f{i+1}": v for i, v in enumerate(image_descriptor)}]
        else:
            item_matching[index] = [
                idItem, {f"f{i+1}": 0 for i in range(size_vector)}]
            print(f"{pathFileFeature} error")
            with open("/root/movie_id_poster_missing_MovieTweeting.csv", "a") as out_missing:
                out_missing.write("{}\n".format(pathFileFeature))
        index += 1
    return item_matching


##For MOVIELENS
def load_movielens_data(file_path='u.data', threshold=0, min_movie_ratings=5, min_user_ratings=5, sep="\t"):
    FILE_MOVIE_MISSING_PATH = f"movielens/movie_id_poster_missing_ml-25m.csv"
    
    names = ['userID', 'itemID', 'rating', 'timestamp']
    dataframe = pd.read_csv(file_path, sep=sep, names=names,
                            engine="python")
    movie_missing = pd.read_csv(FILE_MOVIE_MISSING_PATH, names=['itemID'])
    value_lists = movie_missing['itemID'].drop_duplicates(
    ).sort_values().values.tolist()

    dataframe = dataframe[~dataframe['itemID'].isin(value_lists)]
    dataframe = dataframe[dataframe['rating'] >= threshold]

    filter_movies = (dataframe['itemID'].value_counts() > min_movie_ratings)
    filter_movies = filter_movies[filter_movies].index.tolist()

    filter_users = (dataframe['userID'].value_counts() > min_user_ratings)
    filter_users = filter_users[filter_users].index.tolist()

    dataframe = dataframe[(dataframe['itemID'].isin(filter_movies)) & (
        dataframe['userID'].isin(filter_users))]

    return dataframe


def get_train_test_dataset(df):
    df_train = df.groupby('userID').sample(frac=0.75)
    df_test = df.drop(df_train.index)
    return df_train, df_test


def get_train_test_dataset_new_items(df):
    items_set = df["itemID"].drop_duplicates(
    ).sort_values().reset_index()["itemID"]

    num_items = items_set.shape[0]
    test_size = int(0.1 * num_items)

    rd.shuffle(items_set)

    test_items = items_set[-test_size:]
    df_test = df[df["itemID"].isin(test_items)].copy()

    df_train_tmp = df[~df["itemID"].isin(test_items)].copy()

    df_train = df_train_tmp.groupby('userID').sample(frac=0.75)
    df_test_1 = df_train_tmp.drop(df_train.index)

    df_test = pd.concat([df_test, df_test_1])

    return df_train, df_test


def get_image_features(df, isTest=False, size_vector=SIZE_VECTOR):
    __path_feature ,__ext_feature = get_path_ext_feature_movielens(size_vector=size_vector, ext_feature="VGG19_l1")
    items_set = df["itemID"].drop_duplicates(
    ).sort_values().reset_index()["itemID"]
    num_items = items_set.shape[0]
    item_matching = np.zeros((num_items * 1, 2), dtype=object)
    index = 0
    i = None
    for idItem in items_set:
        if (isTest):
            pathFileFeature = f"{__path_feature}/{idItem}.{__ext_feature}"
        else:
            i = rd.randint(0, 4)
            pathFileFeature = f"{__path_feature}/{idItem}_{i}.{__ext_feature}"
        if (os.path.exists(pathFileFeature)):
            with open(pathFileFeature) as in_feature:
                image_descriptor = np.loadtxt(
                    in_feature, delimiter=",", dtype=np.float64)
                item_matching[index] = [
                    idItem, {f"f{i+1}": v for i, v in enumerate(image_descriptor)}]
        else:
            item_matching[index] = [
                idItem, {f"f{i+1}": 0 for i in range(size_vector)}]
            print(f"{pathFileFeature} error")
        index += 1
    return item_matching


def get_image_features_part(df, i=0, isTest=False, size_vector=SIZE_VECTOR):
    __path_feature ,__ext_feature = get_path_ext_feature_movielens(size_vector)
    items_set = df["itemID"].drop_duplicates(
    ).sort_values().reset_index()["itemID"]
    num_items = items_set.shape[0]
    item_matching = np.zeros((num_items * 1, 2), dtype=object)
    index = 0
    for idItem in items_set:
        if (isTest):
            pathFileFeature = f"{__path_feature}/{idItem}.{__ext_feature}"
        else:
            pathFileFeature = f"{__path_feature}/{idItem}_{i}.{__ext_feature}"
        if (os.path.exists(pathFileFeature)):
            with open(pathFileFeature) as in_feature:
                image_descriptor = np.loadtxt(
                    in_feature, delimiter=",", dtype=np.float64)
                item_matching[index] = [
                    idItem, {f"f{i+1}": v for i, v in enumerate(image_descriptor)}]
        else:
            item_matching[index] = [
                idItem, {f"f{i+1}": 0 for i in range(size_vector)}]
            print(f"{pathFileFeature} error")
        index += 1
    return item_matching


def get_image_features_by_item(idItem, size_vector=SIZE_VECTOR):
    __path_feature ,__ext_feature = get_path_ext_feature_movielens(size_vector)
    item_matching = None
    if (os.path.exists(f"{__path_feature}/{idItem}.{__ext_feature}")):
        with open(f"{__path_feature}/{idItem}.{__ext_feature}") as in_feature:
            image_descriptor = np.loadtxt(
                in_feature, delimiter=",", dtype=np.float64)
            item_matching = [
                idItem, {f"f{i+1}": v for i, v in enumerate(image_descriptor)}]
    else:
        item_matching = [
            idItem, {f"f{i+1}": 0 for i in range(size_vector)}]
    return item_matching


def normalize_feature(df, colums, min=1, max=5):
    df[colums] = df[colums].rank(method="max", pct=True) * (max-min) + min


def normalize(df, colums, min=1, max=5):
    df[colums] = df[colums].rank(method="max", pct=True) * (max-min) + min


def build_interaction_matrix(rows, cols, data):
    """
    Build the training matrix (no_users, no_items),
    with ratings >= 4.0 being marked as positive and
    the rest as negative.
    """

    mat = ssp.lil_matrix((rows, cols), dtype=np.float64)

    for uid, iid, rating in data:
        if rating >= 4.0:
            mat[uid, iid] = 1.0
        else:
            mat[uid, iid] = 0

    return mat.tocoo()


def build_positive_data(interactions, weights, threshold=4):
    weights_tmp = weights.tocsr()
    interactions_tmp = interactions.tocsr()
    for uids, iids in zip(interactions.row, interactions.col):
        if (weights_tmp[uids, iids] < threshold):
            interactions_tmp[uids, iids] = 0.0
        else:
            interactions_tmp[uids, iids] = 1.0
    return interactions_tmp.tocoo()
