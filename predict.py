import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import normalize, plot_results


def compute_model(X: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    """
    model computation calculating the prediction
    values for different input
    Args:
        X (ndarray (m,)): Input parameter
        theta0, theta1 (float): parameters calculated during
        model training to get predictions
    Returns
        (np.ndarray (m,)): model predictions
    """
    cm = np.array([])

    for i in range(len(X)):
        cm = np.append(cm, theta0 + X[i] * theta1)

    return cm


def get_params() -> dict:
    """
    reading the computed model parameters from external file
    Returns
        (dict): dict with the parameters for the model
    """
    file_path = "params.json"

    if not os.path.exists(file_path):
        raise AssertionError("Params File does not exist")

    content = ""

    with open(file_path) as f:
        content = f.read()
        f.close()

    params = json.loads(content)

    return params


def load_csv(file_path: str) -> pd.DataFrame:
    """
    loading data from the csv file path given
    Args:
        file_str (str): file_path as str format
    Returns
        df (pd.DataFrame): DataFrame representing the content for the csv file
    """
    if not os.path.exists(file_path):
        raise AssertionError("csv File does not exist")

    df = pd.read_csv(file_path)

    return df


def predict():
    """
    prediction function that has the multiple
    instruction to get the final prediction result
    """
    try:

        if len(sys.argv) != 2:
            raise AssertionError("invalid number of arguments")

        if sys.argv[1].find(".csv") == -1:
            raise AssertionError("invalid file type, please import .csv file")

        df = load_csv(str(sys.argv[1]))

        params = get_params()

        x_norm = normalize(df["km"])

        cm = compute_model(x_norm, params["theta0"], params["theta1"])

        fig, ax = plt.subplots(1)

        plot_results(ax, df["km"], cm, df["price"])

        plt.show()

    except Exception as e:
        print(f"Error Predict --> {e}")


def main():
    """
    program entrypoint
    """
    predict()


if __name__ == "__main__":
    main()
