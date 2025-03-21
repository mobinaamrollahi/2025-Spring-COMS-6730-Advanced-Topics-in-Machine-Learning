import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

data_dir = "./data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.figure(figsize=(8, 8))

    # If y is continuous, discretize it into bins
    unique_labels = np.unique(y)
    if len(unique_labels) > 2:
        bins = np.linspace(y.min(), y.max(), 2)
        y_binned = np.digitize(y, bins)  # Assign bin numbers
        norm = mcolors.BoundaryNorm(bins, 2)
        cmap = plt.get_cmap('viridis', 2)
    else:
        y_binned = y
        norm = None
        cmap = 'viridis'

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_binned, cmap=cmap, edgecolor='k', alpha=0.75, norm=norm)

    # If y is categorical or discretized, create a legend
    if len(unique_labels) <= 2:
        legend_labels = {val: f"y = {val}" for val in unique_labels}
    else:
        legend_labels = {i: f"{bins[i-1]:.2f} - {bins[i]:.2f}" for i in range(1, 2)}

    # Create legend patches
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=scatter.cmap(scatter.norm(label)), label=text)
                      for label, text in legend_labels.items()]
    
    plt.legend(handles=legend_patches, title="Labels", loc="lower left")
    plt.xlim([-1,0])
    plt.ylim([-1,0])
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.title("Feature Visualization with of y")
    plt.grid(True)
    plt.savefig("train_features.png", dpi=300, bbox_inches='tight')
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.figure(figsize=(8, 8))

    # If y is continuous, discretize it into bins
    unique_labels = np.unique(y)
    if len(unique_labels) > 2:
        bins = np.linspace(y.min(), y.max(), 2)
        y_binned = np.digitize(y, bins)  # Assign bin numbers
        norm = mcolors.BoundaryNorm(bins, 2)
        cmap = plt.get_cmap('viridis', 2)
    else:
        y_binned = y
        norm = None
        cmap = 'viridis'

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_binned, cmap=cmap, edgecolor='k', alpha=0.75, norm=norm)

    # If y is categorical or discretized, create a legend
    if len(unique_labels) <= 2:
        legend_labels = {val: f"y = {val}" for val in unique_labels}
    else:
        legend_labels = {i: f"{bins[i-1]:.2f} - {bins[i]:.2f}" for i in range(1, 2)}

    # Create legend patches
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=scatter.cmap(scatter.norm(label)), label=text)
                        for label, text in legend_labels.items()]
    # Decision boundary: solve for x2 in terms of x1
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1_vals = np.linspace(x1_min, x1_max, 100)

    # Compute x2 using the decision boundary equation: W0 + W1*x1 + W2*x2 = 0
    if len(W) == 3:  # Assuming W includes a bias term
        x2_vals = -(W[0] + W[1] * x1_vals) / W[2]
    else:
        x2_vals = - (W[0] * x1_vals) / W[1]

    plt.legend(handles=legend_patches, title="Labels", loc="lower left")
    # Plot decision boundary
    plt.plot(x1_vals, x2_vals, '--k', linewidth=2, label='Decision Boundary')
    plt.xlim([-1,0])
    plt.ylim([-1,0])
    # Labels, legend, and title
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.title("Logistic Regression Decision Boundary")
    plt.grid(True)

    # Save the figure
    plt.savefig("train_result_sigmoid.png", dpi=300, bbox_inches='tight')
    plt.show()
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
	'''This function is used to plot the softmax model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 0,1,2.
		W: An array of shape [n_features, 3].
	
	Returns:
		No return. Save the plot to 'train_result_softmax.*' and include it
		in submission.
	'''
	### YOUR CODE HERE

	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

    # Visualize training data.
    # print(f"Delete Later. In the visualize_features( function. The {train_X[:, 1:3]} is the train_X[:, 1:3].")
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check GD, SGD, BGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_GD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    results = {}
    learning_rates = [0.1, 0.3, 0.5, 5.0]
    max_iters = [50, 100, 150]
    batch_sizes = [50, 100, 150, 200, 250]  # Only needed for BGD
    algorithms = ['GD', 'BGD', 'SGD']

    for method in algorithms:
        for lr in learning_rates:
            for max_iter in max_iters:
                if method == 'BGD':  # Loop over batch sizes ONLY for BGD
                    for batch_size in batch_sizes:
                        logistic_regression_classifier = logistic_regression(learning_rate=lr, max_iter=max_iter)
                        logistic_regression_classifier.fit_BGD(train_X, train_y, batch_size)

                        # Store results including batch_size for BGD
                        results[(method, lr, max_iter, batch_size)] = logistic_regression_classifier.score(train_X, train_y)
                else:
                    # GD and SGD do NOT require batch_size
                    logistic_regression_classifier = logistic_regression(learning_rate=lr, max_iter=max_iter)
                    
                    if method == 'GD':
                        logistic_regression_classifier.fit_GD(train_X, train_y)
                    elif method == 'SGD':
                        logistic_regression_classifier.fit_SGD(train_X, train_y)

                    # Store results without batch_size
                    results[(method, lr, max_iter)] = logistic_regression_classifier.score(train_X, train_y)

    # Find the best hyperparameters
    best_hyperparams = max(results, key=results.get)
    print("Best hyperparameters:", best_hyperparams)
    best_method = best_hyperparams[0]  # First element is the method
    best_lr = best_hyperparams[1]  # Learning rate
    best_max_iter = best_hyperparams[2]  # Max iterations

    # Check if best method is BGD (which includes batch_size)
    if best_method == "BGD":
        best_batch_size = best_hyperparams[3]  # Extract batch size for BGD
        best_logisticR = logistic_regression(learning_rate=best_lr, max_iter=best_max_iter)
        best_logisticR.fit_BGD(train_X, train_y, best_batch_size)  # Train using best BGD params
    else:
        best_logisticR = logistic_regression(learning_rate=best_lr, max_iter=best_max_iter)
        
        if best_method == "GD":
            best_logisticR.fit_GD(train_X, train_y)
        elif best_method == "SGD":
            best_logisticR.fit_SGD(train_X, train_y)

    # Print the best model's performance
    print("Best Logistic Regression for Binary Classification Results:")
    print("Best Logistic Regression Sigmoid weights:", best_logisticR.get_params())
    print("Best Logistic Regression Sigmoid train accuracy: ", best_logisticR.score(train_X, train_y))
    print("Best Logistic Regression Sigmoid validation accuracy: ", best_logisticR.score(valid_X, valid_y))
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE
    visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    raw_test, label_test = load_data(os.path.join(data_dir, test_filename))
    test_X = prepare_X(raw_test)  
    test_y, test_idx = prepare_y(label_test)  
    test_X = test_X[test_idx]
    test_y = test_y[test_idx] 
    test_y = np.where(test_y == 2, -1, 1)    
    data_shape= test_y.shape[0] 
    print("Test Accuracy:", best_logisticR.score(test_X, test_y))
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  BGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE

    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE

    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE

    ### END YOUR CODE






    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE

    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


'''
Explore the training of these two classifiers and monitor the graidents/weights for each step. 
Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
'''
    ### YOUR CODE HERE

    ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
