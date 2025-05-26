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

def test_model(model_path):
    """Test model with new data"""
    test_df = pd.read_csv("data/processed/test.csv")
    train_df = pd.read_csv("data/processed/train.csv")
    
    X_test = test_df[["total_meters", "floor", "floors_count", "rooms_count"]]
    y_test = test_df["price"]
    X_train = train_df[["total_meters", "floor", "floors_count", "rooms_count"]]
    y_train = train_df["price"]
    model = joblib.load(model_path)
    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    logging.info(f"Gradient boost model metrics.")
    logging.info(f"Test model. MSE: {mse:.2f}")
    logging.info(f"Test model. RMSE: {rmse:.2f}")
    logging.info(f"Test model. MAE: {mae:.2f}")
    logging.info(f"Test model. R2 train: {r2_train:.2f}")
    logging.info(f"Test model. R2 test: {r2_test:.2f}")
    


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

    test_model(model_path)
