import pandas as pd
import numpy as np
import cv2 as cv


# Load dataset from CSV and convert to numpy array
def load_and_preprocess_dataset(csv_file_path, window_length):
    # Load dataset from CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract relevant features from DataFrame
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values

    # Prepare input data with the desired window length
    input_data = []
    for i in range(len(data) - window_length + 1):
        input_data.append(data[i:i+window_length])

    # Convert input data to numpy array
    input_data = np.array(input_data)

    return input_data

# Loads .csv dataset with 5 columns: Daily open, daily low, daily high, closing price, trading volume
def load_dataset(dataset_name):

    # Define columns to extract
    columns_to_extract = ["Open", "Close", "Low", "High", "Volume"]

    # Load the dataset and extract specific columns
    data = pd.read_csv(f'datasets/{dataset_name}', usecols=columns_to_extract)

    # Drop rows with any null, empty, or compromised data values
    data = data.dropna()

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





