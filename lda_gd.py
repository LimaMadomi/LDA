import numpy as np
from lda import LDA


class LDAGD(LDA):
    """
    Implement LDA using GD`.
    """

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components)

    def _calculate_discriminants(self, X, y, iterations=1000):
        # Calculate the scatter matrices
        SW, SB = self.calculate_scatter_matrices(X, y)

        # Initialize the orientation matrix W
        d, h = X.shape[1], self.n_components
        W = np.random.rand(d, h)  # Initialization of W, could alternatively use an identity matrix np.eye(d, h)

        # Iterative gradient descent
        for _ in range(iterations):
            # Compute \( Ĵ(W) \)
            WtSWW = W.T.dot(SW.dot(W))
            WtSBW = W.T.dot(SB.dot(W))
            J_W = np.linalg.det(WtSWW) / np.linalg.det(WtSBW)

            # Update W using the gradient descent step
            inv_WtSWW = np.linalg.inv(WtSWW)
            inv_WtSBW = np.linalg.inv(WtSBW)
            gradient = 2 * J_W * (SW.dot(W).dot(inv_WtSWW) - SB.dot(W).dot(inv_WtSBW))
            W -= 0.01 * gradient  # Using a fixed learning rate of 0.01

            # Normalize the column vectors of W using vectorized operation
            W /= np.linalg.norm(W, axis=0)

        # Assign the learned discriminants
        self.linear_discriminants = W

