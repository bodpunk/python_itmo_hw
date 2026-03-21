import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def prepare_data():
    #data/realty_data.csv удалил, т.к. превышает допустимый размер для гита
    incoming_file = pd.read_csv("data/realty_data.csv")

    train = incoming_file[['price', 'total_square', 'floor']].dropna()

    train['price'] = train['price'].astype(int)
    train['total_square'] = train['total_square'].astype(float)
    train['floor'] = train['floor'].astype(int)

    #удалим сначала выбросы по цене, после чего по площади
    price_lower = train['price'].quantile(0.02)
    price_upper = train['price'].quantile(0.98)
    train = train[(train['price'] >= price_lower) & (train['price'] <= price_upper)]

    square_lower = train['total_square'].quantile(0.02)
    square_upper = train['total_square'].quantile(0.98)
    train = train[(train['total_square'] >= square_lower) & (train['total_square'] <= square_upper)]

    return train


def train_model(train):
    X, y = train.drop("price", axis=1), train['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2026, test_size=0.25)


    lr = make_pipeline(StandardScaler(), LinearRegression(positive=True))
    lr.fit(X_train, y_train)

    with open('lr_fitted.pkl', 'wb') as file:
        pickle.dump(lr, file)


def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model