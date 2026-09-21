import seaborn as sns
import pandas as pd
k_precision = [{"k": 5, "type": "feature", "precision": 0.6},
               {"k": 5, "type": "ratings", "precision": 0.45},
               {"k": 6, "type": "feature", "precision": 0.55},
               {"k": 6, "type": "ratings", "precision": 0.43},
               {"k": 7, "type": "feature", "precision": 0.54},
               {"k": 7, "type": "ratings", "precision": 0.41},
               {"k": 8, "type": "feature", "precision": 0.50},
               {"k": 8, "type": "ratings", "precision": 0.40}]
df = pd.DataFrame(k_precision)
ax = sns.lineplot(
    data=df,
    x="k",  y="precision", hue="type", style="type", markers=True, dashes=True
)
ax.set(title='ĐỘ ĐO PRECISION')
fig = ax.get_figure()
fig.savefig("hist/Test.png")
