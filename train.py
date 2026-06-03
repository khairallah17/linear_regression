import json
import os

import numpy as np
import pandas as pd

from plot import plot_results


def write_values(theta0, theta1, mse_history):
    """
    Computing the prediction of a linear model
    Args:
        theta0, theta1 (float): model parametres
        mse_history (np.ndarray(m, )): residuals result set
    """
    file_path = "features.json"

    mode = "w"

    if not os.path.exists(file_path):
        mode = "x"

    s = {"theta0": theta0, "theta1": theta1, "mse": mse_history}
    with open("features.json", mode) as f:
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
    return (x_train - mean) / std, mean, std


def gradient_descent(x_train, y_train, iterations, lr=0.01):
    """
    Computing the prediction of a linear model
    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m.)): Data, m examples
        w,b (scalar)    : model parameters
    Returns
        f_wb (ndarray (m,)): model prediction
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


def main():
    try:
        df = pd.read_csv("./data.csv")

        x_train = np.array(df["km"])
        y_train = np.array(df["price"])

        x_norm, mean, std = normalize(x_train)

        theta0, theta1, mse_history = gradient_descent(x_norm, y_train, 1000)

        y_pred = theta0 + theta1 * x_norm

        plot_results(x_norm, y_pred, y_train)

        write_values(theta0, theta1, mse_history)

        return

    except Exception as error:
        print(f"error: {error}")
        raise


if __name__ == "__main__":
    main()
