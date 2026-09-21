import csv
from datetime import date
import os
import pandas as pd
import cv2
import time
import gc
import numpy as np
import concurrent.futures

from pathlib import Path
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

PATH_ROOT_DATA = "/home/nckh/nckh_code"
PATH_DATA_CSV = f"{PATH_ROOT_DATA}/movielens/ml-25m/movies.csv"
#PATH_ROOT_DATA = "E:/Data_LV_CuaDuong"
FILE_DATASET = f"{PATH_ROOT_DATA}/movielens/dataset_image_movielens_resize.csv"
PATH_IMAGE_FOLDER = f"{PATH_ROOT_DATA}/image_resize/movielens"
PATH_VGG_FEATURE_MAX = f"{PATH_ROOT_DATA}/feature/movielens/VGG19_max"
PATH_VGG_FEATURE_AVG = f"{PATH_ROOT_DATA}/feature/movielens/VGG19_avg"

PATH_VGG_FEATURE_MAX_RESIZE = f"{PATH_ROOT_DATA}/feature/movielens/VGG19_max.finish_resize"
PATH_VGG_FEATURE_AVG_RESIZE = f"{PATH_ROOT_DATA}/feature/movielens/VGG19_avg.finish_resize"


def extract_id_i(path_to_image):
    name_file = os.path.splitext(os.path.basename(path_to_image))[0]
    id = name_file.split("_")
    return id


def extract_feature_VGG19(model, path_image, pool, id):
    im = cv2.imread(path_image)
    im = cv2.resize(im, (224, 224))
    img_data = preprocess_input(np.expand_dims(im.copy(), axis=0))

    VGG19_feature = model.predict(img_data, use_multiprocessing=True, workers=32)

    VGG19_feature = np.array(VGG19_feature)
    f = ",".join([str(v) for v in VGG19_feature.flatten()])
    if pool == "avg":
        with open(f"{PATH_VGG_FEATURE_AVG}/{id}.VGG19", "w") as out_feature:
            out_feature.write("{}\n".format(f))
        with open(PATH_VGG_FEATURE_AVG_RESIZE, "a") as out_id:
            out_id.write("{}\n".format(f"{id}"))
    elif pool == "max":
        with open(f"{PATH_VGG_FEATURE_MAX}/{id}.VGG19", "w") as out_feature:
            out_feature.write("{}\n".format(f))
        with open(PATH_VGG_FEATURE_MAX_RESIZE, "a") as out_id:
            out_id.write("{}\n".format(f"{id}"))

    del im
    del VGG19_feature
    del img_data
    del f

    gc.collect()


def extract_feature_VGG19_dataset(path_to_folder, dfidi, pool="avg"):
    image_label = []
    start = time.time()
    model = VGG19(weights='imagenet', include_top=False, pooling=pool)
    for layer in model.layers:
        layer.trainable = False
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _, ridi in dfidi.iterrows():
            id = ridi["id"]
            path_image = os.path.join(path_to_folder, f"{id}.jpg")
            image_label.append(id)
            executor.submit(extract_feature_VGG19, model,
                            path_image, pool, id)

    print('Process extract VGG19 time: ', time.time() - start)
    return image_label


if __name__ == "__main__":
    create_dataset = False
    movies = pd.read_csv(
        PATH_DATA_CSV, names=['id', 'title', 'genres'], engine='python', delimiter=',',encoding='utf8',header=0)
    if (create_dataset):
        with open(FILE_DATASET, "w") as out_dataset:
            for _, row in movies.iterrows():
                id = row["id"]
                if(os.path.exists(f"{PATH_IMAGE_FOLDER}/{id}.jpg")):
                    out_dataset.write("{}\n".format(f"{id}"))
    else:
        if (not os.path.exists(PATH_VGG_FEATURE_MAX)):
            Path(PATH_VGG_FEATURE_MAX).mkdir(parents=True, exist_ok=True)
        if (not os.path.exists(PATH_VGG_FEATURE_AVG)):
            Path(PATH_VGG_FEATURE_AVG).mkdir(parents=True, exist_ok=True)

        movide_id_avg_processed = pd.DataFrame(columns=["id"])
        movide_id_max_processed = pd.DataFrame(columns=["id"])

        if (os.path.exists(PATH_VGG_FEATURE_AVG_RESIZE)):
            movide_id_avg_processed = pd.read_csv(
                PATH_VGG_FEATURE_AVG_RESIZE, names=['id'], engine='python', delimiter=',')
        if (os.path.exists(PATH_VGG_FEATURE_MAX_RESIZE)):
            movide_id_max_processed = pd.read_csv(
                PATH_VGG_FEATURE_MAX_RESIZE, names=['id'], engine='python', delimiter=',')

        dataset = pd.read_csv(
            FILE_DATASET, names=['id'], engine='python', delimiter=',')
        dataset_avg = dataset[~dataset.id.isin(movide_id_avg_processed.id.values)]

        dataset_max = dataset[~dataset.id.isin(movide_id_max_processed.id.values)]

        label = extract_feature_VGG19_dataset(
            PATH_IMAGE_FOLDER, dataset_max, "max")
        label = extract_feature_VGG19_dataset(
            PATH_IMAGE_FOLDER, dataset_avg, "avg")
