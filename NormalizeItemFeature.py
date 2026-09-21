from cProfile import label
import numpy as np
import os
import pandas as pd
import sklearn.preprocessing
import support_dataset as spDataset

SIZE_VECTOR = 512
MOVIELENS_SIZE = "25m"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = BASE_DIR if os.path.exists(os.path.join(BASE_DIR, "movielens")) else "/home/nckh/nckh_code"
PATH_ROOT_DATA = os.environ.get("PATH_ROOT_DATA", DEFAULT_ROOT)
PATH_FEATURE = f"{PATH_ROOT_DATA}/feature/movielens/VGG19_avg"
EXTENSION_FEATURE = "VGG19"
EXTENSION_FEATURE_OUT = "VGG19_l1"
PATH_FILE_MOVIES = f"{PATH_ROOT_DATA}/movielens/ml-{MOVIELENS_SIZE}/movies.csv"

if __name__ == "__main__":
    
    if not os.path.exists(f"{PATH_FEATURE}_L1"):
        os.mkdir(f"{PATH_FEATURE}_L1")
    names = ["items", "title", "Gener"]
    movies = pd.read_csv(PATH_FILE_MOVIES, sep=",", names=names, encoding="utf8",
                          engine="python", header=0)
    features = []
    label = []
    items_set = movies["items"].drop_duplicates().sort_values().reset_index()[
        "items"]
    for idItem in items_set:
        for i in range(0, 5):
            pathFileFeature = f"{PATH_FEATURE}/{idItem}_{i}.{EXTENSION_FEATURE}"
            if (os.path.exists(pathFileFeature)):
                with open(pathFileFeature) as in_feature:
                    features.append(np.loadtxt(
                        in_feature, delimiter=",", dtype=np.float64))
                label.append({"id": idItem, "i": i})
        pathFileFeature = f"{PATH_FEATURE}/{idItem}.{EXTENSION_FEATURE}"
       
        if (os.path.exists(pathFileFeature)):
            with open(pathFileFeature) as in_feature:
                features.append(np.loadtxt(
                    in_feature, delimiter=",", dtype=np.float64))
            label.append({"id": idItem, "i": None})
    features = sklearn.preprocessing.normalize(features, norm="l1", copy=False)
    for idx, v in enumerate(label):
        f = features[idx]
        f = ",".join([str(v) for v in f])
        if v["i"] == None:
            fileOutput = f"{PATH_FEATURE}_L1/{v['id']}.{EXTENSION_FEATURE_OUT}"
        else:
            fileOutput = f"{PATH_FEATURE}_L1/{v['id']}_{v['i']}.{EXTENSION_FEATURE_OUT}"

        with open(fileOutput, "w") as out_feature:
            out_feature.write("{}\n".format(f))
