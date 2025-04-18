import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(filename):
    """Load a given txt file.

    Args:
        filename: A string.

    Returns:
        raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
        
    """
    data= np.load(filename)
    x= data['x']
    y= data['y']
    # print(f"Delete Later. in the load data function. The {x} is the x data, and {y} is the y data.")
    return x, y

def train_valid_split(raw_data, labels, split_index):
	"""Split the original training data into a new training dataset
	and a validation dataset.
	n_samples = n_train_samples + n_valid_samples

	Args:
		raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
		split_index: An integer.

	"""
	return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]

def prepare_X(raw_X):
    """Extract features from raw_X as required.

    Args:
        raw_X: An array of shape [n_samples, 256].

    Returns:
        X: An array of shape [n_samples, n_features].
    """
    raw_image = raw_X.reshape((-1, 16, 16))
    # print(f"Delete Later. In the prepare_X function. The {raw_X} is the raw x data, and {raw_image} is the raw_image data.")

	# Feature 1: Measure of Symmetry
	### YOUR CODE HERE
    f_symmetry = -np.sum(np.abs(raw_image - np.flip(raw_image, axis=2)), axis=(1, 2))/256
    # print(f"Delete Later. In the prepare_X function. The {f_symmetry} is the F_symmetry.")
	### END YOUR CODE

	# Feature 2: Measure of Intensity
	### YOUR CODE HERE
    f_intensity = np.sum(raw_image, axis=(1, 2))/256
    # print(f"Delete Later. In the prepare_X function. The {f_intensity} is the F_intensity.")
	### END YOUR CODE

	# Feature 3: Bias Term. Always 1.
	### YOUR CODE HERE
    f0 = np.ones(np.shape(f_symmetry))
    # print(f"Delete Later. In the prepare_X function. The {f0} is the F0.")
	### END YOUR CODE

	# Stack features together in the following order.
	# [Feature 3, Feature 1, Feature 2]
	### YOUR CODE HERE
    X = np.stack([f0, f_symmetry, f_intensity], axis=1)
    # print(f"Delete Later. In the prepare_X function. The {X} is the X stack.")
	### END YOUR CODE
    return X

def prepare_y(raw_y):
    """
    Args:
        raw_y: An array of shape [n_samples,].
        
    Returns:
        y: An array of shape [n_samples,].
        idx:return idx for data label 1 and 2.
    """
    y = raw_y
    idx = np.where((raw_y==1) | (raw_y==2))
    y[np.where(raw_y==0)] = 0
    y[np.where(raw_y==1)] = 1
    y[np.where(raw_y==2)] = 2

    return y, idx




