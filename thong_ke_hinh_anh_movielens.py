import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

names = ['userID', 'itemID', 'rating', 'timestamp']
df = pd.read_csv("MovieTweeting/ratings.dat", sep="::",
                 engine="python", names=names)
max = {"width": 0, "height": 0}
min = None
co_hinh = 0
khong_co_hinh = 0
for iid in df.itemID.unique():
    path_image = f"/root/Data_LV_CuaDuong/MovieTweeting/image_orgin/{iid}.jpg"
    if (os.path.exists(path_image)):
        co_hinh+=1
    else:
        khong_co_hinh+=1
print(co_hinh, khong_co_hinh)
#         img = cv2.imread(path_image)
#         if max["height"] * max["width"] < img.shape[0] * img.shape[1]:
#             max["height"] = img.shape[0]
#             max["width"] = img.shape[1]
#         if min is not None and min["height"] * min["width"] > img.shape[0] * img.shape[1]:
#             min = {"width": img.shape[1], "height": img.shape[0]}
#         else:
#             min = {"width": img.shape[1], "height": img.shape[0]}
# print(max)