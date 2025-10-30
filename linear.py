import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_model(x, y, w, b):
    '''
        Computing the prediction of a linear model
        Args:
            x (ndarray (m,)): Data, m examples
            y (ndarray (m.)): Data, m examples
            w,b (scalar)    : model parameters
        Returns
            f_wb (ndarray (m,)): model prediction
    '''

    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        # print(f"fw(x) = w * {x[i]} = {f_wb[i]}")
        # print(f"{f_wb[i]} += {y[i]} = {f_wb[i] - y[i]}")


    return f_wb

def cost_function(x_train, y_train, iterations, learning_rate):

    '''
        Computing cost function at single iteration
    '''

    compute = compute_model(x_train, y_train, 0.1, 100)
    computed_mean = 0
    computed_margin_error = 0
    
    for i in range(len(x_train)):
        computed_margin_error += (compute[i] - y_train[i])**2
    
    computed_mean = computed_margin_error / len(x_train)

    return cf

def calculte_thetas(x_train, y_train, w, b, itr, lr=0.3):

    '''
        test doc
    '''

    cf = compute_model(x_train, y_train, w, b)

    theta_tmp0 = lr * (1 * cf / len(x_train))

    return [t1, t2]

def main():
    try :
        df = pd.read_csv("./data.csv")

        x_train = np.array(df["km"])
        y_train = np.array(df["price"])

        # tmp_f_wb = compute_model(x_train, y_train, w, b)

        # length = len(tmp_f_wb)

        # marging_error = np.zeros(length)
        # print(length)

        # for i in range(length) :
        #     marging_error[i] = tmp_f_wb[i] - y_train[i]

        cf = cost_function(x_train, y_train, 100, 0.1)
        # return
        # print(cf)

        cf_df = pd.DataFrame(cf)

        x = cf_df["weight"]
        y = cf_df["prediction"]

        # print(cf)

        plt.plot(x, y, c='b', label='Our Predicton')
        # plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
        # plt.title("km price")
        plt.ylabel("weight")
        plt.xlabel("prediction")

        # plt.legend()
        plt.show()

        # df_create = pd.DataFrame({
        #     "km": x_train,
        #     "price": y_train,
        #     "predictions": tmp_f_wb,
        #     "marging_error": marging_error 
        # })

        # df_create.to_csv('predictions.csv')

    except Exception as error:
        print(f"error: {error}")

if __name__ == "__main__":
    main()
