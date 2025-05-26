"""This is full life cycle for ml model"""

import argparse

import pandas as pd
import glob 
import matplotlib.pyplot as plt
import numpy as np
import datetime

import os
import cianparser

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



from sklearn.ensemble import GradientBoostingRegressor


import logging
import joblib


N_ROOMS = 1
TEST_SIZE = 0.2
MODEL_NAME = "gradient_boost_v1.pkl"


# https://docs.python.org/3/library/logging.html

logging.basicConfig(
    filename="./logs/train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)





def train_model(model_path):
    """Train model and save with MODEL_NAME"""
    train_df = pd.read_csv("data/processed/train.csv")
    X = train_df[["total_meters", "floor", "floors_count", "rooms_count"]]  # обучение по 4 признакам
    y = train_df["price"]
    model = GradientBoostingRegressor()
    model.fit(X, y)

    joblib.dump(model, model_path)

    # logging.info(f"Train model. Total meters coef: {model.coef_[0]:.2f}")
    # logging.info(f"Train model. Bias: {model.intercept_:.2f}")




if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-s",
        "--test_train_split",
        type=float,
        help="Split data to test and train dataframes, test size, from 0 to 0.5",
        default=TEST_SIZE,
    )
    
    parser.add_argument(
        "-n", 
        "--n_rooms", 
        help="Number of rooms to parse", 
        type=int, 
        default=N_ROOMS
    )
    
    parser.add_argument(
        "-m", 
        "--model", 
        help="Model name", 
        default=MODEL_NAME
        )
    
    parser.add_argument(
        "-p", 
        "--parse_data", 
        help="Flag to parse new data", 
        action="store_true",
        default=False
    )
    
    args = parser.parse_args()

    test_size = float(args.test_train_split)
    assert 0.0 <= test_size <= 0.5
    model_path = os.path.join("models", args.model)

    train_model(model_path)
