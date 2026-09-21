import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
MOVIELENS_SIZE = "100k"
names = ["index", "k", "type", "precisions", "recall", "f1"]
df = pd.read_csv("result_evaluation/Movielens_100k.csv", header=0, names=names)
df.drop(columns=["index"], inplace=True)

ax = sns.lineplot(
    data=df,
    x="recall",  y="precisions", hue="type", style="type", markers=True, dashes=False,
)

ax.set(title='ĐỘ ĐO PRECISION-RECALL')

plt.xticks([x for x in np.arange(0.6, 1.0, 0.1)])
plt.yticks([x for x in np.arange(0.6, 1.0, 0.1)])
plt.savefig(f"evaluation/precision_recall_Movielens_{MOVIELENS_SIZE}.png")
plt.close()

ax = sns.lineplot(
    data=df,
    x="k",  y="precisions", hue="type", style="type", markers=True, dashes=False
)
ax.set(title='ĐỘ ĐO PRECISION')
fig = ax.get_figure()
fig.savefig(f"evaluation/precision_Movielens_{MOVIELENS_SIZE}.png")
plt.close()

ax = sns.lineplot(
    data=df,
    x="k",  y="recall", hue="type", style="type", markers=True, dashes=True
)
ax.set(title='ĐỘ ĐO RECALL')
fig = ax.get_figure()
fig.savefig(f"evaluation/recall_Movielens_{MOVIELENS_SIZE}.png")
plt.close()

ax = sns.lineplot(
    data=df,
    x="k",  y="f1", hue="type", style="type", markers=True, dashes=True
)
ax.set(title='ĐỘ ĐO F1')
fig = ax.get_figure()
fig.savefig(f"evaluation/f1_Movielens_{MOVIELENS_SIZE}.png")
plt.close()
