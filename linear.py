import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_model(x, w, b):
    '''
        Computing the prediction of a linear model
        Args:
            x (ndarray (m,)): Data, m examples
            w,b (scalar)    : model parameters
        Returns
            f_wb (ndarray (m,)): model prediction
    '''
    
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

def main():
    try :
        df = pd.read_csv("./data.csv")
        print(df.head())

        x_train = np.array(df[" km"])
        y_train = np.array(df["price"]) 

        #plt.show()

        w = 400
        b = 400

        tmp_f_wb = compute_model(x_train, w, b)

        #plt.plot(x_train, tmp_f_wb, c='b', label='Our Predicton')
        plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

        plt.title("km price")
        plt.ylabel("price")
        plt.xlabel("km")

        plt.legend()
        plt.show()

    except Exception as error:
        print(f"error: {error}")

if __name__ == "__main__":
    main()
