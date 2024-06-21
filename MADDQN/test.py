import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from process_data import load_dataset, pandas_to_numpy

# Define the Input Block of the network with 1D convolution
def create_input_block(input_data, input_shape, num_filters, kernel_size):
    """
    Parameters:
    - input_data: numpy array, the input data to be fed into the placeholder.
    - input_shape: tuple, the shape of the input tensor (batch size, window length for time series extraction, number of features (usually 5)).
    - num_filters: int, the number of filters in the convolutional layer.
    - kernel_size: int, the size of the convolutional kernel.

    Returns:
    - inputs_placeholder: TensorFlow placeholder, with input data set.
    - conv_output: TensorFlow tensor, output of the convolutional layer.
    """

    # Define the placeholder for input features
    inputs_placeholder = tf.placeholder(tf.float32, shape=input_shape, name='input_features')

    # Check if the input_data shape matches the expected shape of the placeholder
    expected_shape = inputs_placeholder.get_shape().as_list()
    if input_data.shape != tuple(expected_shape):
        print(input_data.shape)
        print(tuple(expected_shape))
        raise ValueError("Input data shape does not match the expected shape of the placeholder.")

    # Set the inputs of the placeholder object
    inputs_placeholder = tf.placeholder_with_default(input_data, shape=expected_shape, name='input_data')

    # Apply 1D convolutional layer
    conv_output = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation='relu')(inputs_placeholder)

    return inputs_placeholder, conv_output

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

# Example usage:
csv_file_path = "datasets/AAPL.csv"
window_length = 10  # Example window length
input_shape = (4019, window_length, 5)  # Shape of input tensor
num_filters = 16  # Number of filters in the convolutional layer
kernel_size = 3  # Size of the convolutional kernel

# Load and preprocess dataset
input_data = load_and_preprocess_dataset(csv_file_path, window_length)

# Create input block with 1D convolution
inputs_placeholder, conv_output = create_input_block(input_data, input_shape, num_filters, kernel_size)

# Initialize TensorFlow session
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Evaluate the input placeholder tensor and convolutional output tensor
    input_values, conv_output_values = sess.run([inputs_placeholder, conv_output])

    # print("Input tensor values:")
    # print(input_values)
    print("Convolutional output tensor values:")
    print(conv_output_values)
