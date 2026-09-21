import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import support_dataset as spDataset
# MIN_USER_RATING = 50
names = ['userID', 'itemID', 'rating', 'timestamp']
df = spDataset.load_movielens_data(
        "movielens/ml-100k/ratings.dat", sep="::", min_user_ratings=50, min_movie_ratings=5)

user_ratings = df.groupby(['userID'])['itemID'].count().reset_index().sort_values(["itemID"])
item_ratings = df.groupby(['itemID'])['userID'].count().reset_index().sort_values(["userID"])

sns.barplot(data=item_ratings, x="itemID", y="userID")
plt.ylabel("Số lượt đánh giá")
plt.title("SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI ITEM")
plt.savefig("hist/item_ratings_MovieTweeting_filter.png")


sns.barplot(data=user_ratings, x="userID", y="itemID")
plt.ylabel("Số lượt đánh giá")
plt.title("SỐ LƯỢT ĐÁNH GIÁ TƯƠNG ỨNG CHO MỖI USER")
plt.savefig("hist/user_ratings_MovieTweeting_filter.png")
