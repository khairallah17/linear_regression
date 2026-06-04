import matplotlib
import numpy as np


def plot_errors_convergence(ax: matplotlib.axes.Axes, mse: list[float]):
    """
    plotting errors until convergence using mean squared errors
    stored values during the training
    Args:
        ax (matplotlib.axes.Axes): input values
        mse (list[float]): mean sqaured error values
    """
    ax.plot(np.arange(0, 1000, 1), mse)
    ax.set_xlabel("epochs")
    ax.set_ylabel("mean squared error")

    return


def plot_results(
    ax: matplotlib.axes.Axes,
    X: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray
):
    """
    plotting the line results after model training with scatter projection
    of the actual values
    Args:
        ax (matplotlib.axes.Axes): input values
        mse (list[float]): mean sqaured error values
    """
    ax.plot(X, y_pred, label="prediction")
    ax.scatter(X, y_train, c="r", label="errors")

    ax.set_xlabel("mileage")
    ax.set_ylabel("price")

    return
