from tensorflow.keras import datasets
import numpy as np
import pandas as pd
import scipy.io
import random

def MNIST_data(iid = True, shuffle = False):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train / 255.0

    if iid == False:
        x_train_1 = []
        y_train_1 = []
        x_train_2 = []
        y_train_2 = []

        for i in range(60000):
            if y_train[i] <= 4: #30000
                x_train_1.append(x_train[i])
                y_train_1.append(y_train[i])
            else : #30000
                x_train_2.append(x_train[i])
                y_train_2.append(y_train[i])
        
        # K=100
        x_train = np.concatenate([x_train_1[0:80], x_train_2[0:20]])
        y_train = np.concatenate([y_train_1[0:80], y_train_2[0:20]])
        for i in range(1, 375):
            x_train = np.concatenate([x_train, x_train_1[80*(i):80*(i+1)], x_train_2[20*(i):20*(i+1)]])
            y_train = np.concatenate([y_train, y_train_1[80*(i):80*(i+1)], y_train_2[20*(i):20*(i+1)]])
    
    if shuffle == True:
        tmp = list(zip(x_train, y_train))
        random.shuffle(tmp)
        x_train, y_train = zip(*tmp)

    return x_train, y_train, -1

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

def Room_data():
    trainData = pd.read_csv("dataset/Occupancy_Estimation.csv")
    x_train = trainData.iloc[:,:14]
    x_train = x_train.values
    y_train = trainData.iloc[:,14]
    y_train = y_train.array
    input_size = len(x_train[0])
    
    return x_train, y_train, input_size

def Cond_data():
    dataset = scipy.io.loadmat('./dataset/Conductivity.mat')
    X = dataset['X']
    y = dataset['y']
    X = X.tolist()
    y_train = np.array(y).reshape(len(y))
    T = len(y)
    input_size = len(X)
    X_train = [list(i) for i in zip(*X)]
    X_train = np.array(X_train).reshape(T, input_size, 1)
    
    return X_train, y_train, input_size

def Tom_data():
    dataset = scipy.io.loadmat('./dataset/TomData.mat')
    X = dataset['X']
    y = dataset['y']
    X = X.tolist()
    y_train = np.array(y).reshape(len(y))
    T = len(y)
    input_size = len(X)
    X_train = [list(i) for i in zip(*X)]
    X_train = np.array(X_train).reshape(T, input_size, 1)
    
    return X_train, y_train, input_size


def Energy_data():
    dataset = scipy.io.loadmat('./dataset/EnergyData.mat')
    X = dataset['X']
    y = dataset['y']
    X = X.tolist()
    y_train = np.array(y).reshape(len(y))
    T = len(y)
    input_size = len(X)
    X_train = [list(i) for i in zip(*X)]
    X_train = np.array(X_train).reshape(T, input_size, 1)
    
    return X_train, y_train, input_size