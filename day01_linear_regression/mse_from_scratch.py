#library
import numpy as np
from sklearn.metrics import mean_squared_error

#caculate mse
def caculate_mse(y, y_hat):
    e = y_hat-y
    return np.sum(e*e)/len(y)
if __name__ == "__main__":
    y = np.array([10.0, 20.0])
    y_hat = np.array([12.0, 21.0])
    print(caculate_mse(y,y_hat))
    print(mean_squared_error(y,y_hat))