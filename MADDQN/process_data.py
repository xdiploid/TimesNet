import pandas as pd
import numpy as np
import cv2 as cv

# Loads .csv dataset with 5 columns: Daily open, daily low, daily high, closing price, trading volume
def load_dataset(dataset_name):

    data = pd.read_csv(dataset_name)

    # Drop rows with any null, empty, or compromised data values
    data = data.dropna()

    # Returns data as Pandas dataframe
    return data

def pandas_to_numpy(dataset):
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Input data must be a Pandas Dataframe")
    
    numpy_array = dataset.values
    return numpy_array


# Data is split into normalized training and testing sets
def split_train_test_normalize(data, train_ratio=0.8):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a Pandas DataFrame.")

    if not 0 < train_ratio < 1:
        raise ValueError("Train ratio must be between 0 and 1.")

    # Calculate the index for splitting
    split_index = int(train_ratio * len(data))

    # Split the data
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Calculate mean and standard deviation from training data
    train_mean = train_data.mean()
    train_std = train_data.std()

    # Normalize both training and testing data using training statistics
    train_data_normalized = (train_data - train_mean) / train_std
    test_data_normalized = (test_data - train_mean) / train_std

    return train_data_normalized, test_data_normalized





