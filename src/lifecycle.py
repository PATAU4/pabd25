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



from sklearn.ensemble import RandomForestRegressor


import logging
import joblib


N_ROOMS = 1
TEST_SIZE = 0.2
MODEL_NAME = "random_forest_v1.pkl"


# https://docs.python.org/3/library/logging.html

logging.basicConfig(
    filename="./logs/train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def parse_cian(n_rooms=1):
    """Parse data to data/raw"""
    
    moscow_parser = cianparser.CianParser(location="Москва")
    
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
  
    csv_path = f'data/raw/{n_rooms}_{t}.csv'
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 5,
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)

    df.to_csv(csv_path,
              encoding='utf-8',
              index=False)
    
    
    logging.info(f"Raw data parsed. Shape: {df.shape}")



def preprocess_data(test_size):
    """
    Filter, sort and remove duplicates
    """
    raw_data_path = "./data/raw"
    file_list = glob.glob(raw_data_path + "/*.csv")
    
    logging.info(f"Preprocess_data. Use files to train: {file_list}")
    
    main_df = pd.read_csv(file_list[0])
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i])
        df = pd.DataFrame(data)
        main_df = pd.concat([main_df, df], axis=0)

    main_df["url_id"] = main_df["url"].map(lambda x: x.split("/")[-2])
    main_df = main_df[["url_id", "total_meters", "price", "floor", "floors_count", "rooms_count"]].set_index("url_id")
    main_df = main_df.sort_index()
    main_df.drop_duplicates(inplace=True)
    main_df = main_df[main_df["price"] < 100_000_000]
    main_df = main_df[main_df["total_meters"] < 100]
    
    
    train_df, test_df = train_test_split(main_df, test_size=test_size, shuffle=False)

    logging.info(f"Preprocess_data. train_df: {len(train_df)} samples")
    train_head = "\n" + str(train_df.head())
    logging.info(train_head)
    logging.info(f"Preprocess_data. test_df: {len(test_df)} samples")
    test_head = "\n" + str(test_df.head())
    logging.info(test_head)

    train_df.to_csv("data/processed/train.csv")
    test_df.to_csv("data/processed/test.csv")


def train_model(model_path):
    """Train model and save with MODEL_NAME"""
    train_df = pd.read_csv("data/processed/train.csv")
    X = train_df[["total_meters", "floor", "floors_count", "rooms_count"]]  # обучение по 4 признакам
    y = train_df["price"]
    model = RandomForestRegressor(max_depth=12)
    model.fit(X, y)

    joblib.dump(model, model_path)

    # logging.info(f"Train model. Total meters coef: {model.coef_[0]:.2f}")
    # logging.info(f"Train model. Bias: {model.intercept_:.2f}")


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

    logging.info(f"Random forest model metrics.")
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

    # parse_data по умолчанию False
    if args.parse_data:
        parse_cian(args.n_rooms)
        
    preprocess_data(test_size)
    train_model(model_path)
    test_model(model_path)
