import seaborn as sns
import os
import random
import pickle
from pathlib import Path
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import auc_score
# from sklearn.metrics import roc_auc_score
import support_metric as sp
import support_dataset as spDataset
import matplotlib.pyplot as plt
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k, auc, logloss

SIZE_VECTOR = 512
MOVIELENS_SIZE = "10m"
MIN_USER_RATING = 5
FEATURE_NAME = "SIFT" if SIZE_VECTOR == 64 else "VGG19"

PATH_FOLDER_DATASET = "dataset"
PATH_FILE_RATINGS = f"movielens/ml-{MOVIELENS_SIZE}/ratings.dat"
PATH_FILE_LIGHTFM_MODEL = f"model/lightfm_with_feature({FEATURE_NAME})_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).model"


PATH_TRAIN_DATASET = f"{PATH_FOLDER_DATASET}/train_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).pd"
PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).pd"


def predict_user_item(model, user_ids, item_ids, new_item_feature):
    return model.predict(
        user_ids, item_ids=[item_ids], item_features=new_item_feature)[0]

if __name__ == "__main__":
    
    env = dict({
        "FEATURE_NAME": FEATURE_NAME,
        "MOVIELENS_SIZE": MOVIELENS_SIZE,
        "MIN_USER_RATING": MIN_USER_RATING,
        "isNew": False
    })

    ratings = spDataset.load_movielens_data(
       file_path=PATH_FILE_RATINGS, sep="::", min_user_ratings=MIN_USER_RATING)

    if (os.path.exists(PATH_TRAIN_DATASET) and os.path.exists(PATH_TEST_DATASET)):
        ratings_train = pd.read_pickle(PATH_TRAIN_DATASET)
        ratings_test = pd.read_pickle(PATH_TEST_DATASET)
    else:
        ratings_train, ratings_test = spDataset.get_train_test_dataset(
            ratings)
        Path(PATH_FOLDER_DATASET).mkdir(parents=True, exist_ok=True)
        ratings_train.to_pickle(PATH_TRAIN_DATASET)
        ratings_test.to_pickle(PATH_TEST_DATASET)

    dataset = Dataset()
    dataset.fit(ratings['userID'].unique(),
                ratings['itemID'].unique(),
                item_features=[f"f{i+1}" for i in range(SIZE_VECTOR)])

    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, Num items {}.'.format(num_users, num_items))

    (interactions_test, weights_test) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                   for _, x in ratings_test.iterrows())
    item_matching_test = spDataset.get_image_features(
        ratings, isTest=True, size_vector=SIZE_VECTOR)
    item_features_test = dataset.build_item_features(((x[0], x[1])
                                                      for x in item_matching_test), normalize=True)

    (interactions_train, weights_train) = dataset.build_interactions((x['userID'], x['itemID'], x['rating'])
                                                                     for _, x in ratings_train.iterrows())
    user_id_mapping, _, item_id_mapping, _ = dataset.mapping()
    ThuNghiem = False
    item_matching_train = spDataset.get_image_features(ratings_train, isTest=True, size_vector=SIZE_VECTOR)
    item_features_train = dataset.build_item_features(((x[0], x[1])
                                                            for x in item_matching_train), normalize=True)
    if (os.path.exists(PATH_FILE_LIGHTFM_MODEL) and not ThuNghiem):
        model = pickle.load(open(PATH_FILE_LIGHTFM_MODEL, 'rb'))
    else:
        model = LightFM(loss='warp', item_alpha=0.00001)
        #item_matching_train = spDataset.get_image_features(ratings_train, isTest=True, size_vector=SIZE_VECTOR)
        #item_features_train = dataset.build_item_features(((x[0], x[1])
        #                                                    for x in item_matching_train), normalize=True)
        model.fit_partial(interactions_train,
                            item_features=item_features_train, epochs=100, verbose=True, num_threads=32)

        pickle.dump(model, open(PATH_FILE_LIGHTFM_MODEL, 'wb'))
    auc = auc_score(model, interactions_train, item_features=item_features_train)
    print("AUC:", auc)
    evaluation = []
    print("Predict all user-item and Compute K") 
    x = range(1, 21)
    for k in x:
        print(f"Tinh K: {k}")
        df_test = ratings_test[['userID', 'itemID', 'rating']].copy()

        eval = sp.precision_recall_f1_at_k(ratings, user_id_mapping, item_id_mapping,
                                                    df_test=df_test,
                                                    model=model, env=env, item_features=item_features_test,
                                                    num_threads=32, k = k)
        
        # print([{"k": k, "evaluation": "precisions", "value": eval["p"]}, {
        #                   "k": k, "evaluation": "recall", "value": eval["r"]}, {"k": k, "evaluation": "f1", "value": eval["f1"]}])
        evaluation.extend([{"k": k, "evaluation": "precisions", "value": eval["p"]}, {
                          "k": k, "evaluation": "recall", "value": eval["r"]}, {"k": k, "evaluation": "f1", "value": eval["f1"]}])
     
    df_eval = pd.DataFrame(evaluation)
    df_eval.to_csv(
        f"result_evaluation/fm_feature({FEATURE_NAME})_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).csv", index=False)
    plt.ylim(0, 1.0)
    ax = sns.lineplot(
        data=df_eval,
        x="k",  y="value", style="evaluation", hue="evaluation", markers=True, dashes=False
    )
    ax.set(title='ĐÁNH GIÁ MÔ HÌNH FM WITH FEATURE')
    ax.set(xlabel='TOP_K', ylabel='Persent')
    fig = ax.get_figure()
    name_fig = f"evaluation/fm_feature({FEATURE_NAME})_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png"
    print(name_fig)
    fig.savefig(name_fig)
