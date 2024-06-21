import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# To be used on input_data that comes out of the Input Block
def conv_1d(input_data, num_filters, kernel_size):
    """
    Parameters:
    - input_data: TensorFlow object, coming from output of the Input Block
    - num_filters: number of filters to be used in the convolution, default to 16
    - kernel_size: size of kernel to be used for the convolution, default to 3 

    Returns:
    - conv_output: TensorFlow object that contains the data after applying a 1D-convolution
    """

    # Apply 1D convolutional layer to data that comes out of InputBlock
    conv_output = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation='relu')(input_data)

    return conv_output
