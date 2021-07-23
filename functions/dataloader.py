import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class dataloader():
    def __init__(self,path,dataset,test_size):
        print("------------dataloader------------")
        self.test_size = test_size
        if dataset.endswith(".dat"):
            self.ratings_df = pd.read_csv(os.path.join(path,dataset), sep='::', header=None, engine='python')
        else : self.ratings_df = pd.read_csv(os.path.join(path,dataset), encoding = 'utf-8')
        self.ratings_df.columns = ["userId","movieId","rating","timestamp"]
        print("유저 수 :", len(self.ratings_df.userId.unique()))
        print("아이템 수 :", len(self.ratings_df.movieId.unique()))
        self.num_user = len(self.ratings_df.userId.unique())
        self.num_item = len(self.ratings_df.movieId.unique())

    def make_user_autorec_input(self):
        user_item_df = self.ratings_df.pivot_table(values="rating", index="userId", columns="movieId")
        user_item_df.fillna(0,inplace=True)
        self.user_item_df = np.array(user_item_df)
        train_df,test_df = train_test_split(self.user_item_df,test_size =self.test_size)
        return train_df,test_df

    def make_item_autorec_input(self):
        item_user_df = self.ratings_df.pivot_table(values="rating", index="movieId", columns="userId")
        item_user_df.fillna(0,inplace=True)
        self.item_user_df = np.array(item_user_df)
        train_df,test_df = train_test_split(self.item_user_df,test_size =self.test_size)
        return train_df,test_df
