import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_GD(self, X, y):
        """Train perceptron model on data (X,y) with GD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape
		### YOUR CODE HERE
        self.W = np.zeros((n_features,))
        # print("in the fit_GD _x shape:", X.shape)
        # print("in the fit_GD y shape:", y.shape)
        # print("in the fit_GD W shape:", self.W.shape)
        gradient_norm = np.inf  
        tol=1e-6
        k = 0
        while gradient_norm > tol:  
            _x = X[k % n_samples]  
            _y = y[k % n_samples]  

            gradient = self._gradient(_x, _y)

            gradient_norm = np.linalg.norm(gradient, ord=2)

            self.W -= self.learning_rate * gradient
            k += 1 

        ### END YOUR CODE
        return self

    def fit_BGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        k = 0

        while k < self.max_iter:
            k += 1
            # print(f"In the BGD. The iteration is {k}")
            # Randomly select a subset of samples (mini-batch)
            start_idx = np.random.randint(0, n_samples - batch_size + 1)
            end_idx = start_idx + batch_size
            X_S = X[start_idx:end_idx]
            y_S = y[start_idx:end_idx]

            # Compute the logistic regression gradient
            sigmoid = 1 / (1 + np.exp(-np.dot(X_S, self.W)))  
            gradient = np.dot(X_S.T, (sigmoid - y_S)) / batch_size 
            
            # Update the weights
            self.W -= self.learning_rate * gradient
		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with SGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape

        self.W = np.zeros(n_features)
        k = 0

        while k < self.max_iter:
            k += 1
            
            # Randomly choose a sample index
            i = np.random.randint(0, n_samples)
            x_i = X[i]
            y_i = y[i]

            # Compute its contribution to the gradient
            g_i = (np.dot(self.W, x_i) - y_i)* x_i
            # print(f"In the SGD. The Iteration is {k} and the gradient for the sample is {g_i}.")

            # Update the weights
            self.W -= self.learning_rate * g_i

		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        # print("_x shape and _x:", _x)
        # print("y shape and y:", _y)
        # print("W shape and W:", self.W)
        _g = -_y*(1/(1 + np.exp(_y*np.dot(self.W,_x))))*_x
        return _g
		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        exp_term = np.exp(-np.dot(X, self.W))
        preds_proba = np.vstack([1 / (1 + exp_term), exp_term / (1 + exp_term)]).T
        return preds_proba
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        preds = np.sign(np.dot(X, self.W))
        return preds
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        preds = self.predict(X)
        accuracy = np.round(np.mean(preds == y)*100, 2)
        return accuracy
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

