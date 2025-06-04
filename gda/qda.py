import numpy as np

class QDA:

    """
    Quadratic Discriminant Analysis.
    
    This classifier models each class with its own mean and covariance matrix.
    Decision boundaries are quadratic due to class-specific covariance matrices.
    
    Attributes:
        phi (dict): Class priors.
        mu (dict): Class means.
        sigma (dict): Dictionary mapping class labels to class-specific covariance matrices.
        classes (np.ndarray): Array of unique class labels.
    """

    def __init__(self):
        self.phi = []
        self.mu = []
        self.sigma = []
        self.n_classes = None
    
    def fit(self, X, y):

        """
        Fit the QDA model using training data X and labels y.

        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
        """

        self.n_classes = np.unique(y)
        for i in range(self.n_classes.shape[0]):
            phi_i = np.sum(y == i) / len(y)
            self.phi.append(phi_i)
            mu_i = np.mean(X[y == i], axis=0)
            self.mu.append(mu_i)
            sigma_i = (X[y == i] - mu_i).T @ (X[y == i] - mu_i) / len(y)
            self.sigma.append(sigma_i)

    def predict(self, X):

        """
        Predict class labels for input data X.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels.
        """

        pred = []
        m = X.shape[0]
        for i in range(m):
            x = X[i]
            discriminants = []
            for j in range(self.n_classes.shape[0]):
                discriminant = -0.5 * (x - self.mu[j]) @ np.linalg.inv(self.sigma[j] + 1e-8 * np.eye(self.sigma[j].shape[0])) @ (x - self.mu[j]).T + np.log(self.phi[j])
                discriminants.append(discriminant)
            pred.append(np.argmax(discriminants))
        return np.array(pred)
    
