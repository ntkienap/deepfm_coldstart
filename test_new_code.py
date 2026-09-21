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

SIZE_VECTOR = 64
MOVIELENS_SIZE = "100k"
PATH_ROOT_DATA = "/root/Data_LV_CuaDuong/"
PATH_FEATURE = f"{PATH_ROOT_DATA}/feature/Bow_{SIZE_VECTOR}"
EXTENSION_FEATURE = "bow"
FILE_MOVIE_MISSING_PATH = f"movielens/movie_id_poster_missing_1m.csv"

PATH_FOLDER_DATASET = "dataset"
PATH_FILE_RATINGS = f"movielens/ml-{MOVIELENS_SIZE}/ratings.dat"
PATH_FILE_LIGTHFM_MODEL = f"model/ligthfm_with_ratings_new_item_ml_{MOVIELENS_SIZE}.model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_new_item_ml_{MOVIELENS_SIZE}.pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_new_item_ml_{MOVIELENS_SIZE}.pd"

def predict_user_item(model, user_ids, item_ids):
    return model.predict(
        user_ids, item_ids=[item_ids])[0]
    
if __name__ == "__main__":
    ratings = spDataset.load_movielens_data(PATH_FILE_RATINGS, sep="::")
    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset_new_items(ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)

    dataset = Dataset()
    dataset.fit(ratings_train['users'].unique(),
                ratings_train['items'].unique())
    new_iid = ratings_train['items'].max() + 1
    dataset.fit_partial(items=[new_iid])
    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))

    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    (interactions_train, weights_train) = dataset.build_interactions((x['users'], x['items'], x['ratings'])
                                                                     for _, x in ratings_train.iterrows())
    weights_train = weights_train.tocsr()
    train = interactions_train.tocsr()
    for uids, iids in zip(interactions_train.row, interactions_train.col):
        if(weights_train[uids, iids] < 4):
            train[uids, iids] = 0
    interactions_train = train.tocoo()
    
    # if (os.path.exists(PATH_FILE_LIGTHFM_MODEL)):
    #     model = pickle.load(open(PATH_FILE_LIGTHFM_MODEL, 'rb'))
    # else:
    model = LightFM(loss='bpr', no_components=72, user_alpha=5e-3, item_alpha=5e-3,
                    learning_rate=0.005)
    # model.item_biases = 0
    # model.user_biases = 0
    model.fit(interactions_train, sample_weight=weights_train.tocoo(), 
                epochs=300, verbose=True, num_threads=32)
    pickle.dump(model, open(PATH_FILE_LIGTHFM_MODEL, 'wb'))
    print("Start Predict")  
    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    test_set = ratings_test.copy().reset_index()
    user_ids_test = []
    items_ids_test = []
    y_pred = []
    for idx, row in test_set.iterrows():
        uids = user_id_mapping.get(row["users"])
        iids = item_id_mapping.get(row["items"])
        if (iids == None):
            iids = item_id_mapping.get(new_iid)
        y_pred.append(predict_user_item(model, uids, iids))
        
    print("Tinh K")
    dict_pred = [{"UserId": row["users"],
                  "MovieId": row["items"],
                  "Rating": row["ratings"],
                  "Prediction": y_pred[idx],
                  "detail":""}
                 for idx, row in test_set.iterrows()]
    df_pred = pd.DataFrame(dict_pred)
    
    evaluation = []
    x = range(1, 10)
    for k in x:
        p, r = sp.precision_recall_at_k(df_pred.values, k=k, threshold=4)
        f1 = (2 * p * r) / (p + r)
        print(f"Precision@{k}: {p}, Recall@{k}: {r}, F1@{k}: {f1}")
        evaluation.extend([{"k": k, "evaluation": "precisions", "value": p}, {
                          "k": k, "evaluation": "recall", "value": r}, {"k": k, "evaluation": "f1", "value": f1}])
    df = pd.DataFrame(evaluation)
    plt.ylim(0, 1)
    ax = sns.lineplot(
        data=df,
        x="k",  y="value", style="evaluation", hue="evaluation", markers=True, dashes=False
    )
    ax.set(title='ĐÁNH GIÁ MÔ HÌNH MF WITH RATINGS')
    ax.set(xlabel='TOP_K', ylabel='Persent')
    fig = ax.get_figure()
    name_fig = f"evaluation/FM_evaluation_ratings_movilens_{MOVIELENS_SIZE}_new_items.png"
    print(name_fig)
    fig.savefig(name_fig)
