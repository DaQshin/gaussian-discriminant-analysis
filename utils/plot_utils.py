import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plot the decision boundary of a classifier on 2D data.

    Parameters:
        model (object): Trained classifier with a `predict` method.
        X (np.ndarray): Feature matrix of shape (n_samples, 2).
        y (np.ndarray): Ground truth labels.
        title (str): Plot title.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=20)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
