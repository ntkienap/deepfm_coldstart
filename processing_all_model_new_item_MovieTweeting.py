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
import concurrent.futures
import seaborn as sns

SIZE_VECTOR = 64
MOVILENS_SIZE = "1m"
PATH_ROOT_DATA = "/root/Data_LV_CuaDuong/MovieTweeting"
PATH_FEATURE = f"{PATH_ROOT_DATA}/feature/Bow_{SIZE_VECTOR}"
EXTENSION_FEATURE = "bow"
FILE_MOVIE_MISSING_PATH = f"MovieTweeting/movie_id_poster_missing_1m.csv"

PATH_FOLDER_DATASET = "dataset"
PATH_FILE_RATINGS = "MovieTweeting/ratings.dat"
PATH_FILE_LIGTHFM_MODEL_FEATURE = f"model/ligthfm_with_feature_new_item_MovieTweeting_BoW.model"
PATH_FILE_LIGTHFM_MODEL_RATING = f"model/ligthfm_with_raings_new_item_MovieTweeting.model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_new_item_MovieTweeting.pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_new_item_MovieTweeting.pd"

if __name__ == "__main__":

    ratings = spload_movielens_data(PATH_FILE_RATINGS, sep="::")
    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = get_train_test_dataset(ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)

    dataset = Dataset()
    dataset.fit(ratings_train['users'].unique(),
                ratings_train['items'].unique(),
                item_features=[f"f{i+1}" for i in range(SIZE_VECTOR)])
    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))

    item_matching_train = get_image_features(ratings_train)
    item_matching_test = get_image_features(ratings_test)
    (interactions_train, weights_train) = dataset.build_interactions((x['users'], x['items'], x['ratings'])
                                                                     for _, x in ratings_train.iterrows())
    model_feature = pickle.load(
        open(PATH_FILE_LIGTHFM_MODEL_FEATURE, 'rb'))
    model_ratings = pickle.load(
        open(PATH_FILE_LIGTHFM_MODEL_RATING, 'rb'))

    cold_start_ratings = ratings_train["ratings"].mean()
    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    userIds = {y: x for x, y in user_id_mapping.items()}
    itemIds = {y: x for x, y in item_id_mapping.items()}
    y_pre_feature = {}
    y_pre_ratings = {}
    future_excute = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _, row in ratings_test.iterrows():
            userid = row["users"]
            itemid = row["items"]
            item_ids = item_id_mapping.get(itemid)
            user_ids = user_id_mapping.get(userid)     
            if (item_ids == None):
                y_pre_ratings[(userid, itemid)] = [cold_start_ratings]
                dataset.fit_partial(
                    items=[num_items+1], item_features=(f"f{i+1}" for i in range(SIZE_VECTOR)))
                item_ids = dataset.mapping()[2].get(num_items+1)
                new_item_feature = get_image_features_by_item(itemid)
                new_item_feature[0] = num_items+1
                new_item_feature = dataset.build_item_features(
                    [new_item_feature], normalize=False)
                future_excute[executor.submit(
                    predict_new_item, model_feature, user_ids, item_ids, new_item_feature)] = [userid, itemid]
            else:
                y_pre_ratings[(userid, itemid)] = model_ratings.predict(
                    user_ids, item_ids=[item_ids])
                new_item_feature = get_image_features_by_item(itemid)
                new_item_feature = dataset.build_item_features(
                    [new_item_feature], normalize=False)
                future_excute[executor.submit(
                    predict_new_item, model_feature, user_ids, item_ids, new_item_feature)] = [userid, itemid]
        for future in concurrent.futures.as_completed(future_excute):
            (userid, itemid) = future_excute[future]
            y_pre_feature[(userid, itemid)] = future.result()

    dict_pred_feature = [{"UserId": row["users"],
                          "MovieId": row["items"],
                          "Rating": row["ratings"],
                          "Prediction": y_pre_feature[(row["users"], row["items"])][0],
                          "detail":""}
                         for _, row in ratings_test.iterrows()]

    dict_pred_ratings = [{"UserId": row["users"],
                          "MovieId": row["items"],
                          "Rating": row["ratings"],
                          "Prediction": y_pre_ratings[(row["users"], row["items"])][0],
                          "detail":""}
                         for _, row in ratings_test.iterrows()]
    df_pred_feature = pd.DataFrame(dict_pred_feature)
    df_pred_ratings = pd.DataFrame(dict_pred_ratings)
    normalize(df_pred_feature,"Prediction")
    normalize(df_pred_ratings,"Prediction")
    avg = 0.55
    k_feature_precision = []
    k_feature_recall = []
    k_feature_f1 = []
    k_rating_precision = []
    k_rating_recall = []
    k_rating_f1 = []
    x = range(10, 60)
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
    plt.savefig(f"hist/precision_movielens_{MOVILENS_SIZE}_new_item.png")
    plt.close()

    plt.title("RECALL")
    plt.plot(x, np.array(k_feature_recall))
    plt.plot(x, np.array(k_rating_recall))
    plt.legend(['feature', 'ratings'], loc='lower right')
    plt.savefig(f"hist/recall_movielens_{MOVILENS_SIZE}_new_item.png")
    plt.close()

    plt.title("F1")
    plt.plot(x, np.array(k_feature_f1))
    plt.plot(x, np.array(k_rating_f1))
    plt.legend(['feature', 'ratings'], loc='lower right')
    plt.savefig(f"hist/f1_movielens_{MOVILENS_SIZE}_new_item.png")
    plt.close()
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
import concurrent.futures
import seaborn as sns

SIZE_VECTOR = 64
MOVILENS_SIZE = "1m"
PATH_ROOT_DATA = "/root/Data_LV_CuaDuong/"
PATH_FEATURE = f"{PATH_ROOT_DATA}/feature/Bow_{SIZE_VECTOR}"
EXTENSION_FEATURE = "bow"
FILE_MOVIE_MISSING_PATH = f"movielens/movie_id_poster_missing_1m.csv"

PATH_FOLDER_DATASET = "dataset"
PATH_FILE_RATINGS = "movielens/ml-1m/ratings.dat"
PATH_FILE_LIGTHFM_MODEL_FEATURE = f"model/ligthfm_with_feature_new_item_ml_{MOVILENS_SIZE}_BoW.model"
PATH_FILE_LIGTHFM_MODEL_RATING = f"model/ligthfm_with_raings_new_item_ml_{MOVILENS_SIZE}.model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_new_item_ml_{MOVILENS_SIZE}.pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_new_item_ml_{MOVILENS_SIZE}.pd"


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


def get_train_test_dataset(df):
    items_set = df["items"].drop_duplicates(
    ).sort_values().reset_index()["items"]

    num_items = items_set.shape[0]
    test_size = int(0.1 * num_items)
    train_size = num_items - test_size

    random.shuffle(items_set)

    train_items = items_set[:train_size]
    test_items = items_set[-test_size:]
    df_train = df[df["items"].isin(train_items)].copy()
    df_test = df[df["items"].isin(test_items)].copy()

    rows_index = random.sample(
        range(0, df_train.shape[0]), int(0.15 * df_train.shape[0]))
    rows = df_train.iloc[rows_index].copy()

    df_test = pd.concat([df_test, rows])
    df_train.drop(rows.index, axis=0, inplace=True)
    return df_train, df_test


def get_image_features(df):
    items_set = df["items"].drop_duplicates(
    ).sort_values().reset_index()["items"]
    num_items = items_set.shape[0]
    item_matching = np.zeros((num_items * 1, 2), dtype=object)
    index = 0
    for idItem in items_set:
        i = random.randint(0, 4)
        if (os.path.exists(f"{PATH_FEATURE}/{idItem}_{i}.{EXTENSION_FEATURE}")):
            with open(f"{PATH_FEATURE}/{idItem}_{i}.{EXTENSION_FEATURE}") as in_feature:
                image_descriptor = np.loadtxt(in_feature, delimiter=",")
                item_matching[index] = [
                    idItem, {f"f{i+1}": v for i, v in enumerate(image_descriptor)}]
        else:
            item_matching[index] = [
                idItem, {f"f{i+1}": 0 for i in range(SIZE_VECTOR)}]
            #print(f"{idItem} error")
        index += 1
    return item_matching


def get_image_features_by_item(idItem):
    item_matching = None
    if (os.path.exists(f"{PATH_FEATURE}/{idItem}.{EXTENSION_FEATURE}")):
        with open(f"{PATH_FEATURE}/{idItem}.{EXTENSION_FEATURE}") as in_feature:
            image_descriptor = np.loadtxt(in_feature, delimiter=",")
            item_matching = [
                idItem, {f"f{i+1}": v for i, v in enumerate(image_descriptor)}]
    else:
        item_matching = [
            idItem, {f"f{i+1}": 0 for i in range(SIZE_VECTOR)}]
    return item_matching

def normalize(df, column):
    min = df[column].min()
    max = df[column].max()
    df[column] = (df[column] - min) / (max - min)
    
def predict_new_item(model, user_ids, item_ids, new_item_feature):
    return model.predict(
        user_ids, item_ids=[item_ids], item_features=new_item_feature)
# def get_user_features(df):
#     users = df["users"].drop_duplicates().sort_values().reset_index()["users"]
#     dummies = pd.get_dummies(users, prefix="uid")
#     user_features = np.zeros((users.shape[0], 2), dtype=object)
#     for idx, row in dummies.iterrows():
#         user_features[idx] = [users[idx], row.values.flatten().tolist()]
#     return user_features


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
    dataset.fit(ratings_train['users'].unique(),
                ratings_train['items'].unique(),
                item_features=[f"f{i+1}" for i in range(SIZE_VECTOR)])
    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))

    item_matching_train = get_image_features(ratings_train)
    item_matching_test = get_image_features(ratings_test)
    (interactions_train, weights_train) = dataset.build_interactions((x['users'], x['items'], x['ratings'])
                                                                     for _, x in ratings_train.iterrows())
    model_feature = pickle.load(
        open(PATH_FILE_LIGTHFM_MODEL_FEATURE, 'rb'))
    model_ratings = pickle.load(
        open(PATH_FILE_LIGTHFM_MODEL_RATING, 'rb'))
    ratings_test["ratings"] = (ratings_test["ratings"] - ratings_test["ratings"].min()) / (
        ratings_test["ratings"].max() - ratings_test["ratings"].min())
    cold_start_ratings = ratings_train["ratings"].mean()
    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    userIds = {y: x for x, y in user_id_mapping.items()}
    itemIds = {y: x for x, y in item_id_mapping.items()}
    y_pre_feature = {}
    y_pre_ratings = {}
    future_excute = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _, row in ratings_test.iterrows():
            userid = row["users"]
            itemid = row["items"]
            item_ids = item_id_mapping.get(itemid)
            user_ids = user_id_mapping.get(userid)     
            if (item_ids == None):
                y_pre_ratings[(userid, itemid)] = [cold_start_ratings]
                dataset.fit_partial(
                    items=[num_items+1], item_features=(f"f{i+1}" for i in range(SIZE_VECTOR)))
                item_ids = dataset.mapping()[2].get(num_items+1)
                new_item_feature = get_image_features_by_item(itemid)
                new_item_feature[0] = num_items+1
                new_item_feature = dataset.build_item_features(
                    [new_item_feature], normalize=False)
                future_excute[executor.submit(
                    predict_new_item, model_feature, user_ids, item_ids, new_item_feature)] = [userid, itemid]
            else:
                y_pre_ratings[(userid, itemid)] = model_ratings.predict(
                    user_ids, item_ids=[item_ids])
                new_item_feature = get_image_features_by_item(itemid)
                new_item_feature = dataset.build_item_features(
                    [new_item_feature], normalize=False)
                future_excute[executor.submit(
                    predict_new_item, model_feature, user_ids, item_ids, new_item_feature)] = [userid, itemid]
        for future in concurrent.futures.as_completed(future_excute):
            (userid, itemid) = future_excute[future]
            y_pre_feature[(userid, itemid)] = future.result()

    dict_pred_feature = [{"UserId": row["users"],
                          "MovieId": row["items"],
                          "Rating": row["ratings"],
                          "Prediction": y_pre_feature[(row["users"], row["items"])][0],
                          "detail":""}
                         for _, row in ratings_test.iterrows()]

    dict_pred_ratings = [{"UserId": row["users"],
                          "MovieId": row["items"],
                          "Rating": row["ratings"],
                          "Prediction": y_pre_ratings[(row["users"], row["items"])][0],
                          "detail":""}
                         for _, row in ratings_test.iterrows()]
    df_pred_feature = pd.DataFrame(dict_pred_feature)
    df_pred_ratings = pd.DataFrame(dict_pred_ratings)
    normalize(df_pred_feature,"Prediction")
    normalize(df_pred_ratings,"Prediction")
    avg = 0.55
    k_feature_precision = []
    k_feature_recall = []
    k_feature_f1 = []
    k_rating_precision = []
    k_rating_recall = []
    k_rating_f1 = []
    
    k_precision = []
    k_recall = []
    k_f1 = []
    
    x = range(10, 60)
    for k in x:
        print(f"Top_k@{k}...............")
        p_r = sp.precision_recall_feature_ratings(df_pred_feature.values, df_pred_ratings.values, k=k, threshold=avg)
        k_precision.append(p_r[0])
        k_recall.append(p_r[1])
        k_f1.append(p_r[2])
        # precisions_feature, recalls_feature = sp.precision_recall_at_k(
        #     df_pred_feature.values, k=k, threshold=avg)
        # precisions_ratings, recalls_ratings = sp.precision_recall_at_k(
        #     df_pred_ratings.values, k=k, threshold=avg)
        # p = sum(prec for prec in precisions_feature.values()) / \
        #     len(precisions_feature)
        # r = sum(rec for rec in recalls_feature.values()) / len(recalls_feature)
        # f1 = (2 * p * r) / (p + r)
        # k_feature_precision.append(p)
        # k_feature_recall.append(r)
        # k_feature_f1.append(f1)

        # p = sum(prec for prec in precisions_ratings.values()) / \
        #     len(precisions_ratings)
        # r = sum(rec for rec in recalls_ratings.values()) / len(recalls_ratings)
        # f1 = (2 * p * r) / (p + r)
        # k_rating_precision.append(p)
        # k_rating_recall.append(r)
        # k_rating_f1.append(f1)

    df = pd.DataFrame(k_precision)
    sns.lineplot(
        data=df,
        x="k", y="signal", hue="event", style="event",
        markers=True, dashes=False
    )
        
    plt.title("PRECISION")
    plt.plot(x, np.array(k_feature_precision))
    plt.plot(x, np.array(k_rating_precision))
    plt.legend(['feature', 'ratings'], loc='lower right')
    plt.savefig(f"hist/precision_movielens_{MOVILENS_SIZE}_new_item.png")
    plt.close()

    plt.title("RECALL")
    plt.plot(x, np.array(k_feature_recall))
    plt.plot(x, np.array(k_rating_recall))
    plt.legend(['feature', 'ratings'], loc='lower right')
    plt.savefig(f"hist/recall_movielens_{MOVILENS_SIZE}_new_item.png")
    plt.close()

    plt.title("F1")
    plt.plot(x, np.array(k_feature_f1))
    plt.plot(x, np.array(k_rating_f1))
    plt.legend(['feature', 'ratings'], loc='lower right')
    plt.savefig(f"hist/f1_movielens_{MOVILENS_SIZE}_new_item.png")
    plt.close()
