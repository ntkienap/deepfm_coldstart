import os
import pandas as pd
import random as rd
import numpy as np
import scipy.sparse as ssp

SIZE_VECTOR = 512

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = BASE_DIR if os.path.exists(os.path.join(BASE_DIR, "movielens")) else "/home/nckh/nckh_code"

def get_movielens_paths(size=None):
    """Dynamically resolve file paths and delimiters for MovieLens datasets (25m, 20m, 1m, 100k, etc.)."""
    PATH_ROOT_DATA = os.environ.get("PATH_ROOT_DATA", DEFAULT_ROOT)
    if size is None:
        size = os.environ.get("MOVIELENS_SIZE", "25m")
    size = str(size).lower()

    base_ml_dir = os.path.join(PATH_ROOT_DATA, "movielens")
    ml_size_dir = os.path.join(base_ml_dir, f"ml-{size}")

    if size in ["25m", "20m"]:
        ratings_path = os.path.join(ml_size_dir, "ratings.csv")
        movies_path = os.path.join(ml_size_dir, "movies.csv")
        sep = ","
        has_header = True
        missing_path = os.path.join(base_ml_dir, f"movie_id_poster_missing_ml-{size}.csv")
    elif size in ["1m", "10m100k", "10m"]:
        ratings_path = os.path.join(ml_size_dir, "ratings.dat")
        movies_path = os.path.join(ml_size_dir, "movies.dat")
        sep = "::"
        has_header = False
        missing_path = os.path.join(base_ml_dir, f"movie_id_poster_missing_ml-{size}.csv")
    elif size in ["100k"]:
        r_dat = os.path.join(ml_size_dir, "ratings.dat")
        u_data = os.path.join(ml_size_dir, "u.data")
        if os.path.exists(r_dat):
            ratings_path = r_dat
            sep = "::"
        else:
            ratings_path = u_data
            sep = "\t"
        movies_path = os.path.join(ml_size_dir, "u.item")
        has_header = False
        missing_path = os.path.join(base_ml_dir, f"movie_id_poster_missing_ml-100k.csv")
    else:
        ratings_path = os.path.join(ml_size_dir, "ratings.csv") if os.path.exists(os.path.join(ml_size_dir, "ratings.csv")) else os.path.join(ml_size_dir, "ratings.dat")
        movies_path = os.path.join(ml_size_dir, "movies.csv") if os.path.exists(os.path.join(ml_size_dir, "movies.csv")) else os.path.join(ml_size_dir, "movies.dat")
        sep = "," if ratings_path.endswith(".csv") else "::"
        has_header = ratings_path.endswith(".csv")
        missing_path = os.path.join(base_ml_dir, f"movie_id_poster_missing_ml-{size}.csv")

    return {
        "size": size,
        "ratings_path": ratings_path,
        "movies_path": movies_path,
        "sep": sep,
        "has_header": has_header,
        "missing_path": missing_path
    }

def get_path_ext_feature_movielens(size_vector=SIZE_VECTOR, ext_feature=None, pooling="avg", feature_name=None):
    PATH_ROOT_DATA = os.environ.get("PATH_ROOT_DATA", DEFAULT_ROOT)
    feat_base = os.path.join(PATH_ROOT_DATA, "feature", "movielens")
    if not os.path.exists(feat_base):
        feat_base = os.path.join(BASE_DIR, "feature", "movielens")

    # If feature_name is provided, deduce size_vector
    if feature_name:
        fn = str(feature_name).upper()
        if "RESNET" in fn:
            size_vector = 2048
        elif "VGG" in fn:
            size_vector = 512
        elif "BOW" in fn or "SIFT" in fn:
            size_vector = 64

    if size_vector == 2048:
        l1_dir = os.path.join(feat_base, f"ResNet50_{pooling}_L1")
        __path_feature = l1_dir if os.path.exists(l1_dir) else os.path.join(feat_base, f"ResNet50_{pooling}")
        __ext_feature = ext_feature if ext_feature else ("ResNet50_l1" if os.path.exists(l1_dir) else "ResNet50")
    elif size_vector == 512:
        l1_dir = os.path.join(feat_base, f"VGG19_{pooling}_L1")
        __path_feature = l1_dir if os.path.exists(l1_dir) else os.path.join(feat_base, f"VGG19_{pooling}")
        __ext_feature = ext_feature if ext_feature else ("VGG19_l1" if os.path.exists(l1_dir) else "VGG19")
    else:
        l1_dir = os.path.join(feat_base, f"BoW_{size_vector}_L1")
        __path_feature = l1_dir if os.path.exists(l1_dir) else os.path.join(feat_base, f"BoW_{size_vector}")
        __ext_feature = ext_feature if ext_feature else ("bow_l1" if os.path.exists(l1_dir) else "bow")
    return __path_feature, __ext_feature

def get_path_ext_feature_MovieTweeting(size_vector=SIZE_VECTOR, ext_feature=None, pooling="max"):
    PATH_ROOT_DATA = os.environ.get("PATH_ROOT_DATA", f"{DEFAULT_ROOT}/MovieTweeting")
    feat_base = f"{PATH_ROOT_DATA}/feature/MovieTweeting"
    if size_vector == 2048:
        l1_dir = f"{feat_base}/ResNet50_{pooling}_L1"
        __path_feature = l1_dir if os.path.exists(l1_dir) else f"{feat_base}/ResNet50_{pooling}"
        __ext_feature = ext_feature if ext_feature else ("ResNet50_l1" if os.path.exists(l1_dir) else "ResNet50")
    elif size_vector == 512:
        l1_dir = f"{feat_base}/VGG19_{pooling}_L1"
        __path_feature = l1_dir if os.path.exists(l1_dir) else f"{feat_base}/VGG19_{pooling}"
        __ext_feature = ext_feature if ext_feature else ("VGG19_l1" if os.path.exists(l1_dir) else "VGG19")
    else:
        __path_feature = f"{feat_base}/BoW_{size_vector}"
        __ext_feature = ext_feature if ext_feature else "bow"
    return __path_feature, __ext_feature

def load_movietweeting_data(file_path='MovieTweeting/ratings.dat', threshold=0, min_movie_ratings=1, min_user_ratings=1, sep=None):
    FILE_MOVIE_MISSING_PATH = f"MovieTweeting/movie_id_poster_missing_MovieTweeting.csv"
    if sep is None:
        sep = "\t" if str(file_path).endswith(".data") else ("::" if str(file_path).endswith(".dat") else ",")
    names = ['userID', 'itemID', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep=sep, names=names, engine="python")
    if os.path.exists(FILE_MOVIE_MISSING_PATH):
        movie_missing = pd.read_csv(FILE_MOVIE_MISSING_PATH, names=['itemID'])
        value_lists = movie_missing['itemID'].drop_duplicates().sort_values().values.tolist()
        df = df[~df['itemID'].isin(value_lists)]
    df = df[df['rating'] >= threshold]
    while df['itemID'].value_counts().min() <= min_movie_ratings or df['userID'].value_counts().min() <= min_user_ratings:
        filter_movies = (df['itemID'].value_counts() > min_movie_ratings)
        filter_movies = filter_movies[filter_movies].index.tolist()
        filter_users = (df['userID'].value_counts() > min_user_ratings)
        filter_users = filter_users[filter_users].index.tolist()
        df = df[(df['itemID'].isin(filter_movies)) & (df['userID'].isin(filter_users))]
    return df

def get_image_features_MovieTweeting(df, isTest=False, size_vector=SIZE_VECTOR):
    __path_feature, __ext_feature = get_path_ext_feature_MovieTweeting(size_vector)
    items_set = df["itemID"].drop_duplicates().sort_values().reset_index()["itemID"]
    num_items = items_set.shape[0]
    item_matching = np.zeros((num_items * 1, 2), dtype=object)
    index = 0
    for idItem in items_set:
        if isTest:
            pathFileFeature = f"{__path_feature}/{idItem}.{__ext_feature}"
        else:
            i = rd.randint(0, 4)
            pathFileFeature = f"{__path_feature}/{idItem}_{i}.{__ext_feature}"
        if os.path.exists(pathFileFeature):
            with open(pathFileFeature) as in_feature:
                image_descriptor = np.loadtxt(in_feature, delimiter=",", dtype=np.float64)
                item_matching[index] = [idItem, {f"f{i+1}": v for i, v in enumerate(image_descriptor)}]
        else:
            item_matching[index] = [idItem, {f"f{i+1}": 0 for i in range(size_vector)}]
        index += 1
    return item_matching

## For MOVIELENS
def load_movielens_data(file_path=None, size=None, threshold=0, min_movie_ratings=5, min_user_ratings=5, sep=None):
    """Load MovieLens ratings data for any size (25m, 20m, 1m, 100k, etc.) with automatic format detection."""
    if size is None:
        if file_path:
            for s in ["25m", "20m", "10m100k", "1m", "100k"]:
                if f"ml-{s}" in str(file_path).lower():
                    size = s
                    break
        if size is None:
            size = os.environ.get("MOVIELENS_SIZE", "25m")

    info = get_movielens_paths(size)
    if file_path is None:
        file_path = info["ratings_path"]

    if sep is None:
        if str(file_path).endswith(".csv"):
            sep = ","
            has_header = True
        elif str(file_path).endswith(".dat"):
            sep = "::"
            has_header = False
        elif str(file_path).endswith(".data"):
            sep = "\t"
            has_header = False
        else:
            sep = info["sep"]
            has_header = info["has_header"]
    else:
        has_header = info["has_header"] if str(file_path).endswith(".csv") else False

    names = ['userID', 'itemID', 'rating', 'timestamp']
    if has_header:
        dataframe = pd.read_csv(file_path, sep=sep, engine="python", header=0)
        col_map = {
            dataframe.columns[0]: 'userID',
            dataframe.columns[1]: 'itemID',
            dataframe.columns[2]: 'rating',
            dataframe.columns[3]: 'timestamp'
        }
        dataframe = dataframe.rename(columns=col_map)
    else:
        dataframe = pd.read_csv(file_path, sep=sep, names=names, engine="python")

    dataframe['userID'] = pd.to_numeric(dataframe['userID'], errors='coerce').dropna().astype(int)
    dataframe['itemID'] = pd.to_numeric(dataframe['itemID'], errors='coerce').dropna().astype(int)
    dataframe['rating'] = pd.to_numeric(dataframe['rating'], errors='coerce').dropna().astype(float)

    # Filter out missing posters if file exists
    missing_path = info["missing_path"]
    if not os.path.exists(missing_path):
        fallback_missing = os.path.join(DEFAULT_ROOT, "movielens", "movie_id_poster_missing_ml-25m.csv")
        if os.path.exists(fallback_missing):
            missing_path = fallback_missing

    if os.path.exists(missing_path):
        try:
            movie_missing = pd.read_csv(missing_path, header=None)
            missing_ids = pd.to_numeric(movie_missing.iloc[:, 0], errors='coerce').dropna().astype(int).tolist()
            dataframe = dataframe[~dataframe['itemID'].isin(set(missing_ids))]
        except Exception as e:
            print(f"Warning: Could not filter missing posters from {missing_path}: {e}")

    dataframe = dataframe[dataframe['rating'] >= threshold]

    if min_movie_ratings > 0 or min_user_ratings > 0:
        while True:
            m_counts = dataframe['itemID'].value_counts()
            u_counts = dataframe['userID'].value_counts()
            m_cond = m_counts > min_movie_ratings if min_movie_ratings > 0 else True
            u_cond = u_counts > min_user_ratings if min_user_ratings > 0 else True

            if (min_movie_ratings > 0 and (m_counts <= min_movie_ratings).any()) or \
               (min_user_ratings > 0 and (u_counts <= min_user_ratings).any()):
                filter_movies = m_counts[m_cond].index
                filter_users = u_counts[u_cond].index
                prev_len = len(dataframe)
                dataframe = dataframe[(dataframe['itemID'].isin(filter_movies)) & (dataframe['userID'].isin(filter_users))]
                if len(dataframe) == prev_len:
                    break
            else:
                break

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


def get_image_features(df, isTest=False, size_vector=SIZE_VECTOR, ext_feature=None, pooling="avg", feature_name=None):
    __path_feature, __ext_feature = get_path_ext_feature_movielens(size_vector=size_vector, ext_feature=ext_feature, pooling=pooling, feature_name=feature_name)
    if feature_name:
        fn = str(feature_name).upper()
        if "RESNET" in fn:
            size_vector = 2048
        elif "VGG" in fn:
            size_vector = 512
        elif "BOW" in fn or "SIFT" in fn:
            size_vector = 64
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
        index += 1
    return item_matching


def get_image_features_part(df, i=0, isTest=False, size_vector=SIZE_VECTOR, ext_feature=None, pooling="avg", feature_name=None):
    __path_feature, __ext_feature = get_path_ext_feature_movielens(size_vector=size_vector, ext_feature=ext_feature, pooling=pooling, feature_name=feature_name)
    if feature_name:
        fn = str(feature_name).upper()
        if "RESNET" in fn:
            size_vector = 2048
        elif "VGG" in fn:
            size_vector = 512
        elif "BOW" in fn or "SIFT" in fn:
            size_vector = 64
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
        index += 1
    return item_matching


def get_image_features_by_item(idItem, size_vector=SIZE_VECTOR, ext_feature=None, pooling="avg", feature_name=None):
    __path_feature, __ext_feature = get_path_ext_feature_movielens(size_vector=size_vector, ext_feature=ext_feature, pooling=pooling, feature_name=feature_name)
    if feature_name:
        fn = str(feature_name).upper()
        if "RESNET" in fn:
            size_vector = 2048
        elif "VGG" in fn:
            size_vector = 512
        elif "BOW" in fn or "SIFT" in fn:
            size_vector = 64
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
