import numpy as np

class LDA:
    """
    Gaussian Discriminant Analysis (Linear Discriminant Analysis).
    
    This classifier implements a version of Linear Discriminant Analysis where all classes
    share a common covariance matrix, and each class is modeled with a multivariate Gaussian.
    
    Attributes:
        phi (dict): Class priors.
        mu (dict): Class means.
        sigma (np.ndarray): Shared covariance matrix.
        classes (np.ndarray): Array of unique class labels.
    """

    def __init__(self):
        self.phi = {}
        self.mu = {}
        self.sigma = None

    def fit(self, X, y):

        """
        Fit the LDA model using training data.
        
        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Labels of shape (n_samples,).
        """

        n, d = X.shape
        self.classes = np.unique(y)
        self.sigma = np.zeros((d,d))
        for c in self.classes:
            x_c = X[y == c]
            phi_c = np.sum(y == c) / len(y)
            self.phi[c] = phi_c
            mu_c = np.mean(x_c, axis=0)
            self.mu[c] = mu_c
            self.sigma += (x_c - mu_c).T @ (x_c - mu_c)
        
        self.sigma /= n

    def predict(self, X):

        """
        Predict class labels for given input data.
        
        Parameters:
            X (np.ndarray): Input features of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Predicted class labels.
        """

        pred = []
        n, _ = X.shape
        sigma_inv = np.linalg.inv(self.sigma)
        for i in range(n):
            x = X[i]
            discriminants = {}
            for c in self.classes:
                score = -0.5 * (x - self.mu[c]) @ sigma_inv @ (x - self.mu[c]).T + np.log(self.phi[c])
                discriminants[c] = score
            pred.append(max(discriminants, key=discriminants.get))

        return np.array(pred)
    
