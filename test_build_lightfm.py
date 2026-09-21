import numpy as np
import data
from lightfm import LightFM

import inspect

from sklearn.metrics import roc_auc_score

print(inspect.getsource(data._build_interaction_matrix))
def _build_interaction_matrix(rows, cols, data):
    """
    Build the training matrix (no_users, no_items),
    with ratings >= 4.0 being marked as positive and
    the rest as negative.
    """

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, timestamp in data:
        if rating >= 4.0:
            mat[uid, iid] = 1.0
        else:
            mat[uid, iid] = -1.0

    return mat.tocoo()
train, test = data.get_movielens_data()
model = LightFM(no_components=30)
model.fit(train, epochs=50)

train_predictions = model.predict(train.row,
                                  train.col)