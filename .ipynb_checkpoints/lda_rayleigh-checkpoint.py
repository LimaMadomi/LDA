import numpy as np

from lda import LDA


class LDARayleigh(LDA):
    """
    Implement LDA using the standard method aith Rayleigh quotients.
    """

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)

    def _calculate_discriminants(self, X, y):
       # Assuming calculate_scatter_matrices is implemented in the base class and correctly fills self.means
        SW, SB = self.calculate_scatter_matrices(X, y)

        # Check if SW is singular and apply a small regularization if necessary
        if np.linalg.det(SW) == 0:
            SW += np.eye(SW.shape[0]) * 0.01

        # Solving the generalized eigenvalue problem for the matrix SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(SW).dot(SB))

        # Sort the eigenvectors by descending eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top 'n_components' eigenvectors if n_components is specified
        if self.n_components is not None:
            self.linear_discriminants = eigenvectors[:, :self.n_components]
        else:
            self.linear_discriminants = eigenvectors

        # Ensure that the discriminants are stored for use in transformation and prediction
        self.linear_discriminants = eigenvectors[:, :self.n_components]

