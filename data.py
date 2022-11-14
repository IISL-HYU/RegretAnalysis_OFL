from tensorflow.keras import datasets
import numpy as np
import scipy.io

def MNIST_data(iid = True):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train / 255.0

    if iid == False:
        x_train_copy = x_train.copy()
        y_train_copy = y_train.copy()
        x_train_iid = x_train_copy[0:30000]
        y_train_iid = y_train_copy[0:30000]
        x_train_niid = x_train_copy[30000:60000]
        y_train_niid = y_train_copy[30000:60000]

        for index, val in enumerate(y_train_niid):
            if val > 4 and index > 0:
                y_train_niid[index] = y_train_niid[index - 1]
                x_train_niid[index] = x_train_niid[index - 1]
        
        x_train = np.concatenate([x_train_iid[0:500], x_train_niid[0:500]])
        y_train = np.concatenate([y_train_iid[0:500], y_train_niid[0:500]])
        # K=1000
        for i in range(1, 60):
            x_train = np.concatenate([x_train, x_train_iid[500*(i):500*(i+1)], x_train_niid[500*(i):500*(i+1)]])
            y_train = np.concatenate([y_train, y_train_iid[500*(i):500*(i+1)], y_train_niid[500*(i):500*(i+1)]])

    return x_train, y_train

def Air_data():
    dataset = scipy.io.loadmat('./Data/AirDataL.mat')
    X = dataset['X']
    y = dataset['y']
    X = X.tolist()
    y_train = np.array(y).reshape(len(y))
    T = len(y)
    input_size = len(X)
    X_train = [list(i) for i in zip(*X)]
    X_train = np.array(X_train).reshape(T, input_size, 1)
    
    return X_train, y_train, input_size

def Syn_data():
    return