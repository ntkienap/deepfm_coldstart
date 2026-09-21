import numpy as np
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import support_metric as spMetric


import matplotlib.pyplot as plt

SIZE_VECTOR = 512
MIN_USER_RATING = 5  
FEATURE_NAME = "SIFT" if SIZE_VECTOR == 64 else "VGG16"

PATH_FOLDER_DATASET = "dataset/MovieTweeting"

PATH_TEST_DATASET = f"{PATH_FOLDER_DATASET}/test_new_item.pd"

ratings_test = pd.read_pickle(PATH_TEST_DATASET)

test_df = ratings_test[['userID', 'itemID', 'rating']].copy()
df_feature_predictions_all = pd.read_csv(f"prediction_all/MovieTweeting/fm_feature({FEATURE_NAME})_new_item_MovieTweeting.csv",sep=",")
df_ratings_predictions_all = pd.read_csv(f"prediction_all/MovieTweeting/fm_ratings_new_item_MovieTweeting.csv",sep=",")
df_mf_predictions_all = pd.read_csv(f"prediction_all/MovieTweeting/mf_new_item_MovieTweeting.csv",sep=",")

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
    f"result_evaluation/MovieTweeting/all_model_feature({FEATURE_NAME})_MovieTweeting.csv", index=False)
plt.ylim(0, 1)
ax = sns.lineplot(
    data=df,
    x="recall",  y="precisions", hue="model", style="model", markers=True, dashes=False
)
ax.set(title='ĐỘ ĐO PRECISION - RECALL')

fig = ax.get_figure()
fig.savefig(
    f"evaluation/MovieTweeting/precision_recall_feature({FEATURE_NAME})_new_item_MovieTweeting.png")
plt.close()

plt.ylim(0, 1)
ax = sns.lineplot(
    data=df,
    x="k",  y="precisions", hue="model", style="model", markers=True, dashes=False
)
ax.set(title='ĐỘ ĐO PRECISION')
fig = ax.get_figure()
fig.savefig(
    f"evaluation/MovieTweeting/precision_feature({FEATURE_NAME})_new_item_MovieTweeting.png")
plt.close()

plt.ylim(0, 1)
ax = sns.lineplot(
    data=df,
    x="k",  y="recall", hue="model", style="model", markers=True, dashes=True
)
ax.set(title='ĐỘ ĐO RECALL')
fig = ax.get_figure()
fig.savefig(
    f"evaluation/MovieTweeting/recall_feature({FEATURE_NAME})_new_item_MovieTweeting.png")
plt.close()

plt.ylim(0, 1)
ax = sns.lineplot(
    data=df,
    x="k",  y="f1", hue="model", style="model", markers=True, dashes=True
)
ax.set(title='ĐỘ ĐO F1')
fig = ax.get_figure()
fig.savefig(
    f"evaluation/MovieTweeting/f1_feature({FEATURE_NAME})_new_item_MovieTweeting.png")
plt.close()
