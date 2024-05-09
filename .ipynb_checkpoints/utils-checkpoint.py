import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle


def load_IRIS(test=True):
    iris = datasets.load_iris()
    X, y = shuffle(iris.data, iris.target, random_state=1234)
    # iris.target represents the 3 possible classes of iris
    if test:
        X_train = X[:100, :]
        y_train = y[:100]
        X_test = X[100:, :]
        y_test = y[100:]
        mean_train = np.mean(X_train)
        std_train = np.std(X_train)
        X_train = (X_train - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train
        return X_train, y_train, X_test, y_test
    else:
        return X, y


def train_test_split(X, y, test_size=0.3, normalize=False):
    """ Split the data into train and test sets """

    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]
    if normalize == True:
        mean_train = np.mean(X_train)
        std_train = np.std(X_train)
        X_train = (X_train - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train
        # min_train =  np.min(X_train)
        # max_train = np.max(X_train)
        # X_train = (X_train - max_train) / (max_train- min_train)
        # X_test = (X_test - max_train) / (max_train- min_train)

    return X_train, y_train, X_test, y_test


def compute_accuracy(y_true, y_pred):
    """
    Returns the classification accuracy.
    Inputs:
    ----------
    y_true : True labels for X
    y_pred : Predicted labels for X

    Returns
    -------
    accuracy : float
    """
    # Compute the number of correctly predicted labels
    correct_predictions = np.sum(y_true == y_pred)
    # Calculate the accuracy as the ratio of correctly predicted labels to total labels
    accuracy = correct_predictions / len(y_true)
    return accuracy
    ##################################################################
    # ToDo: compute the accuracy among the true and predicted labels #
    # use only numpy functions                                       #
    ##################################################################

