import os
import cv2
import numpy as np
import pandas as pd
import time
import pickle
import csv
import shutil
from pathlib import Path

import concurrent.futures
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

RESIZE_IMAGE = (256, 256)
NUM_CLUSTERS = 64
PATH_ROOT_DATA = "/root/Data_LV_CuaDuong/MovieTweeting"
PATH_FOLDER_IMAGE_CROP = f"{PATH_ROOT_DATA}/image_crop"
PATH_FOLDER_IMAGE_RESIZE = f"{PATH_ROOT_DATA}/image_resize"
PATH_SIFT_FEATURE = f"{PATH_ROOT_DATA}/feature/sift"
PATH_BOW_FEATURE = f"{PATH_ROOT_DATA}/feature/BoW_{NUM_CLUSTERS}"
PATH_FILE_BOW_MODEL = f"model/bow_dictionary_{NUM_CLUSTERS}_MovieTweeting_crop.pkl"
PATH_FILE_FULL_SIFT = f"{PATH_SIFT_FEATURE}/../full_sift_resize.data"
sift = cv2.SIFT_create()


def getId(path_to_image):
    namefile = os.path.splitext(os.path.basename(path_to_image))[0]
    if ("_" in namefile):
        id, i = namefile.split("_")
        return id, i
    else:
        return namefile, None


def extract_sift(path_to_image, id, i=None):
    if (Path(path_to_image).exists()):
        if (i is None and not Path(f"{PATH_SIFT_FEATURE}/{id}.data").exists()):
            img = cv2.imread(path_to_image)
            resized = cv2.resize(img, RESIZE_IMAGE, interpolation=cv2.INTER_AREA)
            kp, des = sift.detectAndCompute(resized, None)
            if len(kp) < 1:
                des = np.zeros((1, sift.descriptorSize()), np.float32)
            with open(f"{PATH_SIFT_FEATURE}/{id}.data", 'w', newline='') as out_feature:
                writer = csv.writer(out_feature, delimiter=',')
                writer.writerows(des)
            del img
            del des

        elif (i is not None and not Path(f"{PATH_SIFT_FEATURE}/{id}_{i}.data").exists()):
            img = cv2.imread(path_to_image)
            resized = cv2.resize(img, RESIZE_IMAGE, interpolation=cv2.INTER_AREA)
            kp, des = sift.detectAndCompute(resized, None)
            if len(kp) < 1:
                des = np.zeros((1, sift.descriptorSize()), np.float32)
            with open(f"{PATH_SIFT_FEATURE}/{id}_{i}.data", 'w', newline='') as out_feature:
                writer = csv.writer(out_feature, delimiter=',')
                writer.writerows(des)
            del img
            del des


def extract_sift_dataset(path_to_folder, extract=True):
    image_label = []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for image_path_file in os.listdir(path_to_folder):
            path_file_image = os.path.join(path_to_folder, image_path_file)
            id, i = getId(path_file_image)
            image_label.append({"id": id, "i": i})
            if (extract):
                executor.submit(extract_sift, path_file_image, id, i)
                executor.submit(extract_sift, path_file_image, id, None)
    print('Process extract SIFT time:', time.time() - start)
    return image_label


def kmeans_bow(image_descriptors, num_clusters):
    start = time.time()
    kmeans = MiniBatchKMeans(n_clusters=num_clusters,
                             verbose=1).fit(image_descriptors)
    print('Process KMeans time:', time.time() - start)
    return kmeans


def merge_all_sift_file(l):
    with open(f"{PATH_SIFT_FEATURE}/{l['id']}.data", 'r', newline='') as in_feature:
        des = np.loadtxt(in_feature, delimiter=",")
    with open(PATH_FILE_FULL_SIFT, 'a', newline='') as out_feature:
        writer = csv.writer(out_feature, delimiter=',')
        writer.writerows(des)
    del des


def create_features_bow(BoW, num_clusters, id, i=None):
    if ((i is None and not Path(f"{PATH_BOW_FEATURE}/{id}.bow").exists()) or (i is not None and not Path(f"{PATH_BOW_FEATURE}/{id}_{i}.bow").exists())):
        image_descriptor = None
        if (i is None):
            extract_sift(f"{PATH_FOLDER_IMAGE_RESIZE}/{id}.jpg", id)
            if (Path(f"{PATH_SIFT_FEATURE}/{id}.data").exists()):
                with open(f"{PATH_SIFT_FEATURE}/{id}.data") as in_feature:
                    image_descriptor = np.loadtxt(in_feature, delimiter=",")
        else:
            extract_sift(f"{PATH_FOLDER_IMAGE_CROP}/{id}_{i}.jpg", id, i)
            if (Path(f"{PATH_SIFT_FEATURE}/{id}_{i}.data").exists()):
                with open(f"{PATH_SIFT_FEATURE}/{id}_{i}.data") as in_feature:
                    image_descriptor = np.loadtxt(in_feature, delimiter=",")
        if (image_descriptor is not None):
            X_features = []
            histo = np.zeros(num_clusters)
            nkp = np.shape(image_descriptor)[0]
            for d in image_descriptor:
                idx = BoW.predict([d])
                histo[idx] += 1/nkp
            X_features.append(histo)
            if (i is None):
                with open(f"{PATH_BOW_FEATURE}/{id}.bow", 'w', newline='') as out_feature:
                    writer = csv.writer(out_feature, delimiter=',')
                    writer.writerows(X_features)
            else:
                with open(f"{PATH_BOW_FEATURE}/{id}_{i}.bow", 'w', newline='') as out_feature:
                    writer = csv.writer(out_feature, delimiter=',')
                    writer.writerows(X_features)
            del X_features
    if (i is None):
        print(f"{PATH_BOW_FEATURE}/{id}.bow is Finish")
    else:
        print(f"{PATH_BOW_FEATURE}/{id}_{i}.bow is finish")


if __name__ == "__main__":
    merge_file = False
    if (not os.path.exists(PATH_SIFT_FEATURE)):
        Path(f"{PATH_SIFT_FEATURE}").mkdir(parents=True, exist_ok=True)
    if (not os.path.exists(PATH_BOW_FEATURE)):
        Path(f"{PATH_BOW_FEATURE}").mkdir(parents=True, exist_ok=True)
    create_dataset = False
    movies = pd.read_csv(
        "MovieTweeting/movies.dat", names=['id', 'title', 'genres'], engine='python', delimiter='::', encoding='utf8')

    # Extract sift feature and merge all sift into file
    # labels = extract_sift_dataset(PATH_FOLDER_IMAGE_CROP, extract=False)
    # if (merge_file):
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         for l in labels:
    #             executor.submit(merge_all_sift_file, l)
    labels = []
    for _, row in movies.iterrows():
        id = row["id"]
        for i in range(5):
            labels.append({"id": id, "i": i})
        labels.append({"id": id, "i": None})

    if not os.path.isfile(PATH_FILE_BOW_MODEL):
        print("Starting read data")
        X = None
        with open(PATH_FILE_FULL_SIFT, 'r', newline='') as in_feature:
            X = np.loadtxt(in_feature, delimiter=",")
        print("Finish read data")
        BoW = kmeans_bow(X, NUM_CLUSTERS)
        pickle.dump(BoW, open(PATH_FILE_BOW_MODEL, 'wb'))
        del X
    else:
        BoW = pickle.load(open(PATH_FILE_BOW_MODEL, 'rb'))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for id_i in labels:
            print("Processing: ", id_i)
            executor.submit(
                create_features_bow, BoW, NUM_CLUSTERS, id_i['id'])
            executor.submit(
                create_features_bow, BoW, NUM_CLUSTERS, id_i['id'], id_i["i"])
    print("Finish")
