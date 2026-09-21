import pandas as pd
import matplotlib.pyplot as plt
import support_dataset as spDataset

names = ['userID', 'itemID', 'rating', 'timestamp']
dataframe = pd.read_csv("MovieTweeting/ratings.dat", sep="::", names=names, engine="python")

filter_movies = (dataframe['itemID'].value_counts() > 5)
filter_movies = filter_movies[filter_movies].index.tolist()
dataframe = dataframe[dataframe['itemID'].isin(filter_movies)]
                      
filter_users = (dataframe['userID'].value_counts() > 5)
filter_users = filter_users[filter_users].index.tolist()

dataframe = dataframe[dataframe['userID'].isin(filter_users)]


count_user = dataframe["userID"].value_counts()
count_item = dataframe["itemID"].value_counts()

print("min_user_rating: ",count_user.min(),"; max_user_rating: ",count_user.max())
print("min_item_rating: ",count_item.min(),"; max_item_rating: ",count_item.max())
# exit(0)
dataframe["userID"].value_counts().to_csv("hist/MovieTweting_user_rating_count.csv")
dataframe["userID"].value_counts().hist()
plt.savefig("hist/MovieTweting_user_rating_histogram.png")
plt.close()