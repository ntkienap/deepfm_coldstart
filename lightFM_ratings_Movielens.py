import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
import support_metric as sp
import support_dataset as spDataset
import matplotlib.pyplot as plt
import seaborn as sns
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k, auc

SIZE_VECTOR = 512
MOVIELENS_SIZE = "10m"
MIN_USER_RATING = 5
FEATURE_NAME = "SIFT" if SIZE_VECTOR == 64 else "VGG19"

PATH_FOLDER_DATASET = "dataset"
PATH_FILE_RATINGS = f"movielens/ml-{MOVIELENS_SIZE}/ratings.dat"

PATH_FILE_LIGHTFM_MODEL = f"model/lightfm_with_ratings_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).model"

PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).pd"


if __name__ == "__main__":
    env = dict({
        "FEATURE_NAME": "NONE",
        "MOVIELENS_SIZE": MOVIELENS_SIZE,
        "MIN_USER_RATING": MIN_USER_RATING
    })
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
    
    dataset = Dataset()
    dataset.fit(ratings['userID'].unique(),
                ratings['itemID'].unique())

    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))
 
    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    (interactions_train, weights_train) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                     for _, x in ratings_train.iterrows())

    if (os.path.exists(PATH_FILE_LIGHTFM_MODEL)):
        model = pickle.load(open(PATH_FILE_LIGHTFM_MODEL, 'rb'))
    else:
        model = LightFM(no_components=41, loss='warp', learning_rate=0.001,
                        item_alpha=0.00022, user_alpha=0.00033, max_sampled=15)
        model.fit(interactions_train,
                  epochs=100, verbose=True, num_threads=32)
        pickle.dump(model, open(PATH_FILE_LIGHTFM_MODEL, 'wb'))

    print("Predict all user-item")
    print("Predict all user-item and Compute K") 
    test_df = ratings_test[['userID', 'itemID', 'rating']].copy()
    evaluation=[]
    x = range(1, 21)
    for k in x:
        print(f"Tinh K: {k}")
        df_test = ratings_test[['userID', 'itemID', 'rating']].copy()

        eval = sp.precision_recall_f1_at_k(ratings, user_id_mapping, item_id_mapping,
                                                    df_test=df_test,
                                                    model=model, env=env, item_features=None,
                                                    num_threads=32, k = k)
        
        # print([{"k": k, "evaluation": "precisions", "value": eval["p"]}, {
        #                   "k": k, "evaluation": "recall", "value": eval["r"]}, {"k": k, "evaluation": "f1", "value": eval["f1"]}])
        evaluation.extend([{"k": k, "evaluation": "precisions", "value": eval["p"]}, {
                          "k": k, "evaluation": "recall", "value": eval["r"]}, {"k": k, "evaluation": "f1", "value": eval["f1"]}])
     
    df_eval = pd.DataFrame(evaluation)
    df_eval.to_csv(
        f"result_evaluation/fm_ratings_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).csv", index=False)
    plt.ylim(0, 1)
    ax = sns.lineplot(
        data=df_eval,
        x="k",  y="value", style="evaluation", hue="evaluation", markers=True, dashes=False
    )
    ax.set(title='ĐÁNH GIÁ MÔ HÌNH FM WITH RATINGS')
    ax.set(xlabel='TOP_K', ylabel='Persent')
    fig = ax.get_figure()
    name_fig = f"evaluation/fm_ratings_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png"
    print(name_fig)
    fig.savefig(name_fig)
