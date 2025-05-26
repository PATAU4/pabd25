"""This is full life cycle for ml model"""

import argparse

import pandas as pd
import glob 
import matplotlib.pyplot as plt
import numpy as np

import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.ensemble import GradientBoostingRegressor

import logging
import joblib



TEST_SIZE = 0.2



# https://docs.python.org/3/library/logging.html

logging.basicConfig(
    filename="./logs/train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

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

    train_df.to_csv("data/processed_data/train.csv")
    test_df.to_csv("data/processed_data/test.csv")

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
    
    args = parser.parse_args()

    test_size = float(args.test_train_split)
    assert 0.0 <= test_size <= 0.5

        
    preprocess_data(test_size)

