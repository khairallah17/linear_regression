import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot import plot_results


def write_values(theta0, theta1):
    """
    writing the parameters values to extrenal file for
    further testing
    Args:
        theta0, theta1 (float): model parametres
        mse_history (np.ndarray(m, )): residuals result set
    """
    file_path = "params.json"

    mode = "w"

    if not os.path.exists(file_path):
        mode = "x"

    s = {"theta0": theta0, "theta1": theta1}
    with open(file_path, mode) as f:
        json.dump(s, f)
        f.close()

    return


def normalize(x_train):
    """
    Computing the prediction of a linear model
    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m.)): Data, m examples
        w,b (scalar)    : model parameters
    Returns
        f_wb (ndarray (m,)): model prediction
    """
    mean = x_train.mean()
    std = x_train.std()
    return (x_train - mean) / std


def RMSE(rss, size):
    return math.sqrt(rss / size)


def RSS(y_pred, y_train):

    rs = (y_train - y_pred) ** 2
    return np.sum(rs)


def TSS(y_train, mean):

    ts = (y_train - mean) ** 2
    return np.sum(ts)


def evaluate(y_pred, y_train, mse_history):

    rss = RSS(y_pred, y_train)
    tss = TSS(y_train, np.mean(y_train))
    r_square = 1 - (rss / tss)

    rmse = RMSE(rss, y_pred.size)

    return r_square, rmse


def gradient_descent(
    x_train: np.ndarray, y_train: np.ndarray, iterations: int, lr: float = 0.01
):
    """
    calculating the values for model parametres using gradient
    descent search algorithm
    this algorithm works by finding the lowest point of the curve
    it works by find the slope and the intercept for each point of
    the curve until convergence
    Args:
        x_train (ndarray (m,)): input values
        y_train (ndarray (m,)): output values
        iterations (int)    : number of iteration for the algorithm
    Returns
        theta0, theta1 : final model parameters
        mse_histroy: mean squared errors values untill convergence
    """
    theta0 = 0
    theta1 = 0
    mse_history = []
    m = len(y_train)

    for _ in range(iterations):

        y_pred = theta0 + (theta1 * x_train)

        error = y_pred - y_train

        mse = np.mean(error**2) / m
        mse_history.append(mse)

        grad_theta0 = (2 / m) * np.sum(error)
        grad_theta1 = (2 / m) * np.sum(error * x_train)

        theta0 -= lr * grad_theta0
        theta1 -= lr * grad_theta1

    return theta0, theta1, mse_history


def plot_errors_convergence(mse):

    plt.plot(np.arange(0, 1000, 1), mse)
    plt.xlabel("epochs")
    plt.ylabel("mean squared error")
    plt.show()

    return


def train():
    try:
        df = pd.read_csv("./data.csv")

        x_train = np.array(df["km"])
        y_train = np.array(df["price"])

        x_norm = normalize(x_train)

        theta0, theta1, mse_history = gradient_descent(x_norm, y_train, 1000)

        y_pred = theta0 + theta1 * x_norm

        plot_errors_convergence(mse_history)
        plot_results(x_norm, y_pred, y_train)

        r_square, rmse = evaluate(y_pred, y_train, mse_history)

        print(f"r\u00b2 = {r_square*100:.2f}%\nRMSE = {rmse}")

        write_values(theta0, theta1)

        return

    except Exception as error:
        print(f"error: {error}")
        raise


def main():
    train()


if __name__ == "__main__":
    main()
