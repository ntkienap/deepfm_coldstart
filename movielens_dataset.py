import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

names = ['users', 'items', 'ratings', 'timestamp']
dataframe = pd.read_csv("movielens/ml-100k/ratings.dat", sep="::", names=names,
                    engine="python")

dataframe["ratings"].hist()
plt.savefig("hist/movielens-100k.png")