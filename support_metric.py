from multiprocessing import Lock, Process, Queue, current_process, cpu_count, Manager
import time
import queue

import pandas as pd
import numpy as np
from collections import defaultdict


# from sklearn.preprocessing import minmax_scale

from sklearn.metrics import roc_auc_score
from recommenders.utils.spark_utils import start_or_get_spark
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k, auc, logloss


def _get_header():
    COL_USER = "userID"
    COL_ITEM = "itemID"
    COL_RATING = "rating"
    COL_PREDICTION = "prediction"

    HEADER = {
        "col_user": COL_USER,
        "col_item": COL_ITEM,
        "col_rating": COL_RATING,
        "col_prediction": COL_PREDICTION,
    }
    return HEADER


def prepare_dfs(df_true, df_pred):
    spark = start_or_get_spark("EvaluationTesting", "local")
    dfs_true = spark.createDataFrame(df_true)
    dfs_pred = spark.createDataFrame(df_pred)
    return dfs_true, dfs_pred


def precision_recall_model(dfs_true, dfs_pred, k=5, model="VGG19-FM"):
    header = _get_header()
    spark_rank_eval = SparkRankingEvaluation(
        dfs_true, dfs_pred, k=k, relevancy_method="top_k", **header)
    p = spark_rank_eval.precision_at_k()
    r = spark_rank_eval.recall_at_k()
    f1 = (2 * p * r) / (p + r)
    return [{"k": k, "model": f"{model}-FM", "precisions": p, "recall": r, "f1": f1}]


def precision_recall_all_model(df_true, df_feature_prediction, df_ratings_prediction, df_mf_prediction, k=5, feature_name="VGG16"):
    p_f = precision_at_k(rating_true=df_true,
                         rating_pred=df_feature_prediction, k=k)
    r_f = recall_at_k(rating_true=df_true,
                      rating_pred=df_feature_prediction, k=k)

    p_r = precision_at_k(rating_true=df_true,
                         rating_pred=df_ratings_prediction, k=k)
    r_r = recall_at_k(rating_true=df_true,
                      rating_pred=df_ratings_prediction, k=k)

    p_mf = precision_at_k(rating_true=df_true,
                          rating_pred=df_mf_prediction, k=k)
    r_mf = recall_at_k(rating_true=df_true,
                       rating_pred=df_mf_prediction, k=k)

    f1_f = (2 * p_f * r_f) / (p_f + r_f)
    f1_r = (2 * p_r * r_r) / (p_r + r_r)
    f1_mf = (2 * p_mf * r_mf) / (p_mf + r_mf)
    return [{"k": k, "model": f"{feature_name}-FM", "precisions": p_f, "recall": r_f, "f1": f1_f},
            {"k": k, "model": "FM", "precisions": p_r, "recall": r_r, "f1": f1_r},
            {"k": k, "model": "MF", "precisions": p_mf, "recall": r_mf, "f1": f1_mf}]


def precision_recall_all_model_multi_feature(df_true, df_feature_vgg16_prediction, df_feature_sift_prediction, df_ratings_prediction, df_mf_prediction, k=5):
    p_f_vgg16 = precision_at_k(rating_true=df_true,
                               rating_pred=df_feature_vgg16_prediction, k=k)
    r_f_vgg16 = recall_at_k(rating_true=df_true,
                            rating_pred=df_feature_vgg16_prediction, k=k)

    p_f_sift = precision_at_k(rating_true=df_true,
                              rating_pred=df_feature_sift_prediction, k=k)
    r_f_sift = recall_at_k(rating_true=df_true,
                           rating_pred=df_feature_sift_prediction, k=k)

    p_r = precision_at_k(rating_true=df_true,
                         rating_pred=df_ratings_prediction, k=k)
    r_r = recall_at_k(rating_true=df_true,
                      rating_pred=df_ratings_prediction, k=k)

    p_mf = precision_at_k(rating_true=df_true,
                          rating_pred=df_mf_prediction, k=k)
    r_mf = recall_at_k(rating_true=df_true,
                       rating_pred=df_mf_prediction, k=k)

    f1_f_vgg16 = (2 * p_f_vgg16 * r_f_vgg16) / (p_f_vgg16 + r_f_vgg16)
    f1_f_sift = (2 * p_f_sift * r_f_sift) / (p_f_sift + r_f_sift)
    f1_r = (2 * p_r * r_r) / (p_r + r_r)
    f1_mf = (2 * p_mf * r_mf) / (p_mf + r_mf)
    return [{"k": k, "type": "feature(VGG16)", "precisions": p_f_vgg16, "recall": r_f_vgg16, "f1": f1_f_vgg16},
            {"k": k, "type": "feature(Sift)", "precisions": p_f_sift,
             "recall": r_f_sift, "f1": f1_f_sift},
            {"k": k, "type": "ratings", "precisions": p_r, "recall": r_r, "f1": f1_r},
            {"k": k, "type": "mf", "precisions": p_mf, "recall": r_mf, "f1": f1_mf}]


def precision_recall_at_k(predictions, k=10, threshold=4):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = len(user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            true_r >= threshold for (_, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    p = sum(prec for prec in precisions.values()) / \
        len(precisions)
    r = sum(rec for rec in recalls.values()) / len(recalls)
    return p, r


def full_auc(model, ground_truth, feature=None):

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    scores = []

    for user_id, row in enumerate(ground_truth):
        uid_array = np.empty(no_items, dtype=np.int32)
        uid_array.fill(user_id)
        if (feature is None):
            predictions = model.predict(uid_array, pid_array, num_threads=4)
        else:
            predictions = model.predict(
                uid_array, pid_array, item_features=feature, num_threads=4)

        true_pids = row.indices[row.data == 1]

        grnd = np.zeros(no_items, dtype=np.int32)
        grnd[true_pids] = 1

        if len(true_pids):
            scores.append(roc_auc_score(grnd, predictions))

    return sum(scores) / len(scores)


def full_auc_mf(model, ground_truth, feature=None):

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    scores = []

    for user_id, row in enumerate(ground_truth):
        uid_array = np.empty(no_items, dtype=np.int32)
        uid_array.fill(user_id)
        test_index = [(uid, iid, 1) for uid, iid in zip(uid_array, pid_array)]
        predictions = model.test(test_index)
        predictions = [est for _, _, _, est, _ in predictions]
        true_pids = row.indices[row.data == 1]

        grnd = np.zeros(no_items, dtype=np.int32)
        grnd[true_pids] = 1

        if len(true_pids):
            scores.append(roc_auc_score(grnd, predictions))

    return sum(scores) / len(scores)


def normalize(df, column, range=(0, 5), in_replace=True):
    min = df[column].min()
    max = df[column].max()
    if in_replace:
        df[column] = (df[column] - min)/(max - min)
        df[column] = df[column] * (range[1] - range[0]) + range[0]
        return df
    else:
        df_temp = df.copy()
        df_temp[column] = (df_temp[column] - min)/(max - min)
        df_temp[column] = df_temp[column] * (range[1] - range[0]) + range[0]
        return df_temp


def prepare_all_predictions(
    data,
    uid_map,
    iid_map,
    interactions,
    model,
    num_threads,
    user_features=None,
    item_features=None,
):
    users, items, preds = [], [], []  # noqa: F841
    item = list(data.itemID.unique())
    print("call function")
    for user in data.userID.unique():
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)

    all_predictions = pd.DataFrame(data={"userID": users, "itemID": items})

    all_predictions["uid"] = all_predictions.userID.map(uid_map)
    all_predictions["iid"] = all_predictions.itemID.map(iid_map)

    uids = all_predictions["uid"].values
    iids = all_predictions["iid"].values
    all_predictions["prediction"] = None
    print("predict process")
    all_predictions["prediction"] = model.predict(
        user_ids=uids,
        item_ids=iids,
        user_features=user_features,
        item_features=item_features,
        num_threads=num_threads,
    )
    return all_predictions[["userID", "itemID", "prediction"]]


def do_task_evaluation(result_list, predictions, df_test, model, env, num_threads=3, user_features=None, item_features=None, k=3):
    uids = predictions["uid"].values
    iids = predictions["iid"].values
    predictions["prediction"] = model.predict(
        user_ids=uids,
        item_ids=iids,
        user_features=user_features,
        item_features=item_features,
        num_threads=num_threads
    )

    df_pre = predictions[["userID", "itemID", "prediction"]].copy()
    p = precision_at_k(rating_true=df_test,
                       rating_pred=df_pre, k=k)
    r = recall_at_k(rating_true=df_test,
                    rating_pred=df_pre, k=k)

    f1 = 0 if (p + r == 0) else (2 * p * r) / (p + r)
    d = result_list[0]
    d["ps"].append(p)
    d["rs"].append(r)
    d["f1s"].append(f1)
    result_list[0] = d
    if env['isNew']:
        df_pre.to_csv(
            f"prediction_all/fm_feature({env['FEATURE_NAME']})_new_item_ml-{env['MOVIELENS_SIZE']}_UMR({env['MIN_USER_RATING']}).csv", mode='a', index=False, header=False)
    else:
        df_pre.to_csv(
            f"prediction_all/fm_feature({env['FEATURE_NAME']})_ml-{env['MOVIELENS_SIZE']}_UMR({env['MIN_USER_RATING']}).csv", mode='a', index=False, header=False)


def precision_recall_f1_at_k(
    data,
    uid_map,
    iid_map,
    df_test,
    model,
    env,
    num_threads=3,
    user_features=None,
    item_features=None,
    k=3
):
    itemset = list(data.itemID.unique())
    userset = data.userID.unique()
    num_user = len(userset)
    all_predictions = pd.DataFrame(
        columns=['userID', 'itemID', 'uid', 'iid', 'prediction'])
    if env['isNew']:
        all_predictions.to_csv(
            f"prediction_all/fm_feature({env['FEATURE_NAME']})_new_item_ml-{env['MOVIELENS_SIZE']}_UMR({env['MIN_USER_RATING']}).csv", mode='a', index=False)
    else:
        all_predictions.to_csv(
            f"prediction_all/fm_feature({env['FEATURE_NAME']})_ml-{env['MOVIELENS_SIZE']}_UMR({env['MIN_USER_RATING']}).csv", mode='a', index=False)

    users, items = [], []
    count = 0
    processes = []
    manager = Manager()
    result_list = manager.list()
    result_list.append({"ps": [], "rs": [], "f1s": []})
    for user in userset:
        users.extend([user] * len(itemset))
        items.extend(itemset)
        count += 1
        if (count % 1000 == 0 or count == num_user):
            predictions = pd.DataFrame(data={"userID": users, "itemID": items})
            predictions["uid"] = predictions.userID.map(uid_map)
            predictions["iid"] = predictions.itemID.map(iid_map)
            p = Process(target=do_task_evaluation, args=(result_list,
                                                         predictions, df_test, model, env, num_threads, user_features, item_features, k))
            processes.append(p)
            p.start()
            users, items = [], []
    for p in processes:
        p.join()
    eval_dict = result_list[0]
    ps = eval_dict["ps"]
    rs = eval_dict["rs"]
    f1s = eval_dict["f1s"]

    return {"p": sum(ps)/len(ps), "r": sum(rs)/len(rs), "f1": sum(f1s)/len(f1s)}


def prepare_all_predictions_for_mf(
    data,
    data_train,
    model
):
    preds_lst = []
    items = list(data.itemID.unique())
    users = list(data.userID.unique())
    for user in users:
        for item in items:
            preds_lst.append([user, item, model.predict(user, item).est])

    all_predictions = pd.DataFrame(data=preds_lst, columns=[
                                   "userID", "itemID", "prediction"])

    tempdf = pd.concat(
        [
            data_train[["userID", "itemID"]],
            pd.DataFrame(
                data=np.ones(data_train.shape[0]), columns=["dummycol"], index=data_train.index
            ),
        ],
        axis=1,
    )
    merged = pd.merge(tempdf, all_predictions, on=[
                      "userID", "itemID"], how="outer")
    return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)


def predict_to_pandas(data, uid_map, iid_map, model, item_features=None, num_threads=32):
    df_predictions = pd.DataFrame(
        data={"userID": data.userID.values, "itemID": data.itemID.values})
    df_predictions["uid"] = df_predictions.userID.map(uid_map)
    df_predictions["iid"] = df_predictions.itemID.map(iid_map)
    df_predictions["prediction"] = model.predict(
        df_predictions.uid.values, df_predictions.iid.values, item_features=item_features, num_threads=num_threads)
    normalize(df_predictions, ["prediction"])
    return df_predictions[["userID", "itemID", "prediction"]]


def predict_user_item(model, user_ids, item_ids, new_item_feature):
    return model.predict(
        user_ids, item_ids=[item_ids], item_features=new_item_feature)[0]
