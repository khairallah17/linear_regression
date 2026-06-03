import matplotlib.pyplot as plt


def plot_convergence(mse_history):

    plt.plot(mse_history, c="r", label="errors")

    plt.show()
    return


def plot_results(X, y_pred, y_train):

    plt.plot(X, y_pred, label="prediction")
    plt.scatter(X, y_train, c="r", label="errors")

    # plt.scatter(x_norm, y_train, c="r", label="Actual Values")
    # plt.title("km price")
    plt.xlabel("mileage")
    plt.ylabel("price")
    plt.show()

    return
