from tensorflow.keras import datasets
import numpy as np
import scipy.io
import random

def MNIST_data(iid = True, shuffle = False):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train / 255.0

    if iid == False:
        x_train_1 = x_train[0:48000]
        y_train_1 = y_train[0:48000]
        x_train_2 = x_train[48000:54000]
        y_train_2 = y_train[48000:54000]
        x_train_3 = x_train[53999:59999]
        y_train_3 = y_train[53999:59999]

        for i in range(1, 48000):
            if y_train_1[i] > 4:
                x_train_1[i] = x_train_1[i-1]
                y_train_1[i] = y_train_1[i-1]
        for i in range(1, 6000):
            if y_train_2[i] < 5 or y_train_2[i] > 8:
                x_train_2[i] = x_train_2[i-1]
                y_train_2[i] = y_train_2[i-1]
        for i in range(1, 6000):
            if y_train_3[i] < 8:
                x_train_3[i] = x_train_3[i-1]
                y_train_3[i] = y_train_3[i-1]
        
        x_train = np.concatenate([x_train_1[0:800], x_train_2[0:100], x_train_3[0:100]])
        y_train = np.concatenate([y_train_1[0:800], y_train_2[0:100], y_train_3[0:100]])
        # K=1000
        for i in range(1, 60):
            x_train = np.concatenate([x_train, x_train_1[800*(i):800*(i+1)], x_train_2[100*(i):100*(i+1)], x_train_3[100*(i):100*(i+1)]])
            y_train = np.concatenate([y_train, y_train_1[800*(i):800*(i+1)], y_train_2[100*(i):100*(i+1)], y_train_3[100*(i):100*(i+1)]])
    
    if shuffle == True:
        tmp = list(zip(x_train, y_train))
        random.shuffle(tmp)
        x_train, y_train = zip(*tmp)

    return x_train, y_train, 0

def Air_data():
    dataset = scipy.io.loadmat('./dataset/AirDataL.mat')
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