
import os
import random
import pickle
import numpy as np
import pandas as pd
import support_metric as sp
import matplotlib.pyplot as plt
import seaborn as sns
from lightfm.evaluation import auc_score

from pathlib import Path
import support_metric as spMetric
import support_dataset as spDataset
from sklearn.preprocessing import minmax_scale

import copy
import os
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from lightfm.data import Dataset as LightFMDataset
from lightfm import LightFM
from surprise import Dataset as MFDataset, Reader, SVD

import support_metric as sp
import matplotlib.pyplot as plt
import support_dataset as spDataset

SIZE_VECTOR = 512
MOVIELENS_SIZE = "100k"
MIN_USER_RATING = 50
FEATURE_NAME = "SIFT" if SIZE_VECTOR == 64 else "VGG16"

PATH_FOLDER_DATASET = "dataset/"

PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).pd"

ratings_test = pd.read_pickle(PATH_TEST_DATASET)

test_df = ratings_test[['userID', 'itemID', 'rating']].copy()
df_feature_predictions_all = pd.read_csv(f"prediction_all/fm_feature({FEATURE_NAME})_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).csv",sep=",")
df_ratings_predictions_all = pd.read_csv(f"prediction_all/fm_ratings_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).csv",sep=",")
df_mf_predictions_all = pd.read_csv(f"prediction_all/mf_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).csv",sep=",")

evaluation = []
x = range(1, 21)
for k in x:
    print(f"Top_k@{k}...............")
    ev = spMetric.precision_recall_all_model(
        test_df, df_feature_predictions_all, df_ratings_predictions_all, df_mf_predictions_all, k=k,feature_name=FEATURE_NAME)
    print(ev)
    evaluation.extend(ev)

df = pd.DataFrame(evaluation)
df.to_csv(
    f"result_evaluation/all_model_feature({FEATURE_NAME})_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).csv", index=False)
plt.ylim(0, 1)
ax = sns.lineplot(
    data=df,
    x="recall",  y="precisions", hue="model", style="model", markers=True, dashes=False
)
ax.set(title='ĐỘ ĐO PRECISION - RECALL')

fig = ax.get_figure()
fig.savefig(
    f"evaluation/precision_recall_feature({FEATURE_NAME})_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png")
plt.close()

plt.ylim(0, 1)
ax = sns.lineplot(
    data=df,
    x="k",  y="precisions", hue="model", style="model", markers=True, dashes=False
)
ax.set(title='ĐỘ ĐO PRECISION')
fig = ax.get_figure()
fig.savefig(
    f"evaluation/precision_feature({FEATURE_NAME})_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png")
plt.close()

plt.ylim(0, 1)
ax = sns.lineplot(
    data=df,
    x="k",  y="recall", hue="model", style="model", markers=True, dashes=True
)
ax.set(title='ĐỘ ĐO RECALL')
fig = ax.get_figure()
fig.savefig(
    f"evaluation/recall_feature({FEATURE_NAME})_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png")
plt.close()

plt.ylim(0, 1)
ax = sns.lineplot(
    data=df,
    x="k",  y="f1", hue="model", style="model", markers=True, dashes=True
)
ax.set(title='ĐỘ ĐO F1')
fig = ax.get_figure()
fig.savefig(
    f"evaluation/f1_feature({FEATURE_NAME})_new_item_ml-{MOVIELENS_SIZE}_UMR({MIN_USER_RATING}).png")
plt.close()
