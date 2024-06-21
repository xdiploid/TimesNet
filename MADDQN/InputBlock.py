import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from process_data import load_and_preprocess_dataset

# Defines the Input Block of the network
def create_input_block(input_data, input_shape):
    """
    Parameters:
    - input_data: numpy array, the input data to be fed into the placeholder.
    - input_shape: tuple, the shape of the input tensor (batch size, window length for time series extraction, number of features (usually 5)).

    Returns:
    - inputs_placeholder: TensorFlow placeholder, with input data set.
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

    return inputs_placeholder



# Example usage:
# csv_file_path = "datasets/AAPL.csv"
# window_length = 10  # Example window length
# input_shape = (4019, window_length, 5)  # Shape of input tensor

# # Load and preprocess dataset
# input_data = load_and_preprocess_dataset(csv_file_path, window_length)

# # Create input block
# inputs_placeholder = create_input_block(input_data, input_shape)



