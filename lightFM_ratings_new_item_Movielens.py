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
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k, auc, logloss

MOVIELENS_SIZE = "1m"
MIN_USER_RATING = 50

PATH_ROOT_DATA = "/root/Data_LV_CuaDuong"

PATH_FOLDER_DATASET = "dataset"
PATH_FILE_RATINGS = f"movielens/ml-{MOVIELENS_SIZE}/ratings.dat"
PATH_FILE_LIGHTFM_MODEL = f"model/lightfm_with_ratings_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).pd"


def predict_user_item(model, user_ids, item_ids):
    return model.predict(
        user_ids, item_ids=[item_ids])[0]


if __name__ == "__main__":
    ratings = spDataset.load_movielens_data(PATH_FILE_RATINGS, sep="::", min_user_ratings=MIN_USER_RATING)
    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset_new_items(
            ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)
    print(ratings.shape, ratings.userID.unique().shape, ratings.itemID.unique().shape)
    print(ratings_train.shape, ratings_train.userID.unique().shape, ratings_train.itemID.unique().shape)
    print(ratings_test.shape, ratings_test.userID.unique().shape, ratings_test.itemID.unique().shape)
    exit(0)
    
    dataset = Dataset()
    dataset.fit(ratings['userID'].unique(),
                ratings['itemID'].unique())
    print(ratings_train.shape)
    print(ratings_train.userID.unique().shape)
    print(ratings_train.itemID.unique().shape)
    
    print(ratings_test.shape)
    print(ratings_test.userID.unique().shape)
    print(ratings_test.itemID.unique().shape)

    user_ratings_train = ratings_train.groupby(['userID'])['itemID'].count().reset_index().sort_values(["itemID"])
    item_ratings_train = ratings_train.groupby(['itemID'])['userID'].count().reset_index().sort_values(["userID"])
    item_ratings_test = ratings_test.groupby(['itemID'])['userID'].count().reset_index().sort_values(["userID"])
    user_ratings_test = ratings_test.groupby(['userID'])['itemID'].count().reset_index().sort_values(["itemID"])

    print(user_ratings_train.itemID.mean())
    print(user_ratings_train.itemID.min())
    print(user_ratings_train.itemID.max())
    
    print(user_ratings_test.itemID.mean())
    print(user_ratings_test.itemID.min())
    print(user_ratings_test.itemID.max())
    
    
    sns.barplot(data=item_ratings_train, x="itemID", y="userID")
    plt.ylabel("Số lượt đánh giá")
    plt.title("SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI ITEM")
    plt.savefig(f"hist/item_ratings_train_ml-{MOVIELENS_SIZE}_filter.png")
    plt.close()

    sns.barplot(data=user_ratings_train, x="userID", y="itemID")
    plt.ylabel("Số lượt đánh giá")
    plt.title("SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI USER")
    plt.savefig(f"hist/user_ratings_train_ml-{MOVIELENS_SIZE}_filter.png")
    plt.close()
    
    sns.barplot(data=item_ratings_test, x="itemID", y="userID")
    plt.ylabel("Số lượt đánh giá")
    plt.title("SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI ITEM")
    plt.savefig(f"hist/item_ratings_test_ml-{MOVIELENS_SIZE}_filter.png")
    plt.close()

    sns.barplot(data=user_ratings_test, x="userID", y="itemID")
    plt.ylabel("Số lượt đánh giá")
    plt.title("SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI USER")
    plt.savefig(f"hist/user_ratings_test_ml-{MOVIELENS_SIZE}_filter.png")
    plt.close()
    exit(0)
    
    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))

    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    (interactions_train, weights_train) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                     for _, x in ratings_train.iterrows())

    ThuNghiem = True
    if (os.path.exists(PATH_FILE_LIGHTFM_MODEL) and not ThuNghiem):
        model = pickle.load(open(PATH_FILE_LIGHTFM_MODEL, 'rb'))
    else:
        model = LightFM(no_components=41, loss='warp', learning_rate=0.001,
                        item_alpha=0.00022, user_alpha=0.00033, max_sampled=15)

        model.fit(interactions_train,
                  epochs=150, verbose=True, num_threads=32)
        pickle.dump(model, open(PATH_FILE_LIGHTFM_MODEL, 'wb'))

    (interactions_test, weights_test) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                   for _, x in ratings_test.iterrows())

    print("Predict all user-item")
    test_df = ratings_test[['userID','itemID','rating']].copy()
    df_all_predictions = sp.prepare_all_predictions(ratings, user_id_mapping, item_id_mapping, 
                                              interactions=interactions_train,
                                              model=model,
                                              num_threads=32)
    df_all_predictions.to_csv(f"prediction_all/fm_ratings_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).csv",index=False)
    print("Tinh K..........")
    evaluation = []
    x = range(1, 21)
    for k in x:
        p = precision_at_k(rating_true=test_df, 
                            rating_pred=df_all_predictions, k=k)
        r = recall_at_k(test_df, df_all_predictions, k=k)
        f1 = (2 * p * r) / (p + r)
        print(f"Precision@{k}: {p}, Recall@{k}: {r}, F1@{k}: {f1}")
        evaluation.extend([{"k": k, "evaluation": "precisions", "value": p}, {
                          "k": k, "evaluation": "recall", "value": r}, {"k": k, "evaluation": "f1", "value": f1}])
    df_eval = pd.DataFrame(evaluation)
    df_eval.to_csv(f"result_evaluation/fm_ratings_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).csv", index=False)
    plt.ylim(0, 1)
    ax = sns.lineplot(
        data=df_eval,
        x="k",  y="value", style="evaluation", hue="evaluation", markers=True, dashes=False
    )
    ax.set(title='ĐÁNH GIÁ MÔ HÌNH FM WITH RATINGS')
    ax.set(xlabel='TOP_K', ylabel='Persent')
    fig = ax.get_figure()
    name_fig = f"evaluation/fm_ratings_new_items_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png"
    print(name_fig)
    fig.savefig(name_fig)
