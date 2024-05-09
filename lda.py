from telnetlib import SB
from tkinter.constants import SW

import numpy as np
import matplotlib.pyplot as plt


class LDA:
    """
    Base class for an LDA classifier.
    """

    def __init__(self, n_components=None):

        self.n_components = n_components

        # Shape will be (num_features, n_components)
        self.linear_discriminants = None

        # This vector should be filled with the mean vectors for each class in the dataset.
        # Each vector is of shape (num_feature,)
        # and each element if the mean value of the corresponding feature in the dataset.
        # One filled, for the binary classification task, this list will thus hold two elements (one for class), each one with shape (num_features,)
        self.means = []

    def train(self, X, y):
        """Train the model.

        This function should call the appropriate method 
        and set self `self.discriminants`.

        
        Parameters
        ----------
        X : array-like, shape (num_samples, n_features)
            Training vectors, where num_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (num_samples,)
            Target values.

        Returns
        -------
        None
        """

        # if X is None or len(X) == 0:
        #     raise ValueError("Training data cannot be empty.")
        # if y is None or len(y) == 0:
        #     raise ValueError("Labels cannot be empty.")
        # if len(np.unique(y)) != 2:
        #     raise ValueError("LDA requires exactly two classes for binary classification.")
        #
        # SW, SB = self.calculate_scatter_matrices(X, y)
        # self._calculate_discriminants(SW, SB)

        classes = np.unique(y)
        assert len(classes) == 2, "There must be exactly two classes."

        self.means = []
        for cls in classes:
            class_mean = X[y == cls].mean(axis=0)
            self.means.append(class_mean)

        # Continue with computing SW, SB and discriminants
        SW, SB = self.calculate_scatter_matrices(X, y)
        self._calculate_discriminants(SW, SB)
    def calculate_scatter_matrices(self, X, y):
        """

        This function should compute and return the within-class and between-class scatter matrices.

        Suggestion: fill `self.means` in this method.

        Parameters
        ----------
        X : array-like, shape = [num_samples, n_features]
        y : array, shape (num_samples,) containing the target values

        Returns
        -------

        Tuple (SW, SB), where
            SW : array-like, shape: [n_features, n_features]
            SB : array-like, shape: [n_features, n_features]
        """
        n_features = X.shape[1]
        labels = np.unique(y)  # Unique classes in the dataset
        overall_mean = np.mean(X, axis=0)  # Overall mean of all features

        SW = np.zeros((n_features, n_features))  # Within-class scatter matrix
        SB = np.zeros((n_features, n_features))  # Between-class scatter matrix
        self.means = []  # List to store class means

        for label in labels:
            X_class = X[y == label]  # Data points for the current class
            mean_class = np.mean(X_class, axis=0)  # Mean of the current class
            self.means.append(mean_class)  # Append current class mean to self.means

            # Within-class scatter matrix calculation using vectorized operation
            diff = X_class - mean_class  # Difference from the class mean
            SW += diff.T.dot(diff)  # Dot product of the differences

            # Between-class scatter matrix calculation
            mean_diff = mean_class - overall_mean
            n_class_samples = X_class.shape[0]
            SB += n_class_samples * np.outer(mean_diff, mean_diff)  # Outer product

        return SW, SB

    def _calculate_discriminants(self, X, y):
        """

        This function should compute the linear discriminants and assign them to 
        `self.linear_discriminants`.

        The function is not implemented in the `LDA` base class. 
        Different implementations will be provided for each required approach specified in the TP.

        Parameters
        ----------
        X : array-like, shape = [num_samples, n_features]
        y : array, shape (num_samples,) containing the target values

        Returns
        -------
        None
        """
        # Solving the generalized eigenvalue problem for the matrix SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(SW).dot(SB))

        # Sort the eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top 'n_components' eigenvectors
        self.linear_discriminants = eigenvectors[:, :self.n_components]
        raise NotImplementedError()

    def transform(self, X):
        """

        Parameters
        ----------
        X : array-like, shape = [num_samples, n_features]

        Returns
        -------
        predictions : array, shape = [self.n_components]
            Projections of input samples using the linear discriminants in `self.linear_discriminants`.
        """

        assert self.linear_discriminants is not None
        return X @ self.linear_discriminants

    def predict(self, X):
        """
        Returns predictions for binary classification task.

        For the decision boundary, we use a simple heuristic:
        The threshold value is computed by averaging the mean value for each class 
        that you should have computed in `self.means`.

        For binary classificaiton, given class means M1 and M2, compute M = (M1+M2)/2.
        Then, this value is projectd onto the linear discriminants to give the threshold value.

        This threshold should be used to perform the classification decision. 
        That is, for a test point projected onto the linear discriminants, with projection x', 
        x'>threshold gives us the positive class, while x'<= threshold holds the negative class.

        Parameters
        ----------
        X : array-like, shape = [num_samples, n_features]

        Returns
        -------
        predictions : array, shape = [num_samples]
            Predicted target values for X
        """
        # assert self.linear_discriminants is not None
        #
        # # There should only be 2 mean values (one for each class)
        # # as this is binary classificatio.
        # assert len(self.means) == 2
        # # Calculate the projection of the input features onto the discriminant space
        # projections = self.transform(X)
        #
        # # Compute the threshold by projecting the average of the class means onto the discriminant space
        # mean_vector = np.mean(self.means, axis=0)  # Average the means of the two classes
        # threshold = np.dot(mean_vector, self.linear_discriminants)
        #
        # # Compare projections to the threshold to make class predictions
        # return (projections > threshold).astype(int)

        if self.linear_discriminants is None:
            raise ValueError("Linear discriminants have not been computed.")

        if len(self.means) != 2:
            raise ValueError("Expected two class means but got {}".format(len(self.means)))

            # Calculate the projection of the input features onto the discriminant space
        projections = self.transform(X)

        # Compute the threshold by projecting the average of the class means onto the discriminant space
        mean_vector = np.mean(self.means, axis=0)  # Average the means of the two classes
        threshold = np.dot(mean_vector, self.linear_discriminants)

        # Compare projections to the threshold to make class predictions
        return (projections > threshold).astype(int)
    def plot_1d(self, X, y, title=None):
        """ This function plots the projected datapoints to a single line.
        This should be used for datasets with two clases, where `self.n_components == 1`.

        Note: plot the 1D projectios on a single line. You can choose the line where y=0
        to plot the points on. Assign a different color to each class.
        """

        assert self.n_components == 1

        # Transform data
        projections = self.transform(X)
        # Plotting
        plt.figure(figsize=(10, 2))
        for label in np.unique(y):
            plt.scatter(projections[y == label], np.zeros_like(projections[y == label]), label=f'Class {label}')
        plt.title(title if title else '1D LDA Projection')
        plt.xlabel('Projected axis')
        plt.yticks([])
        plt.legend()
        plt.show()

    def plot_2d(self, X, y, title=None):
        """ Plot the dataset X and the corresponding labels y in 2D using the LDA
        transformation.
        Assign a different color to each class.
        """

        if self.n_components != 2:
            raise ValueError("plot_2d function requires n_components to be exactly 2, got {}".format(self.n_components))

            # Transform data
        projections = self.transform(X)

        # Ensure projections have the correct shape
        if projections.shape[1] != 2:
            raise ValueError(
                "Projections should have exactly two columns for 2D plotting, got shape {}".format(projections.shape))

        # Plotting
        plt.figure(figsize=(8, 6))
        for label in np.unique(y):
            plt.scatter(projections[y == label, 0], projections[y == label, 1], label=f'Class {label}', alpha=0.7)
        plt.title(title if title else '2D LDA Projection')
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.legend()
        plt.grid(True)
        plt.show()
