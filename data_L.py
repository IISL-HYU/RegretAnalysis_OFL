import random
import numpy as np
import tensorflow as tf
import emnist

def EMNIST_data():
    print(emnist.list_datasets())
    images, labels = emnist.extract_training_samples('letters')
    print(images.shape)
    images = images.reshape((124800, 28, 28, 1))
    images = images / 255.0

    return 'ENMIST', images, labels, -3


def MNIST_data(iid = True, shuffle = False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train / 255.0

    return 'MNIST', x_train, y_train, -1

def CIFAR_10_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.reshape((50000, 32, 32, 3))
    x_train = x_train / 255.0
    
    return 'CIFAR_10', x_train, y_train, -2


def data_shuffle(x_train, y_train):
    tmp = list(zip(x_train, y_train))
    random.shuffle(tmp)
    x_train, y_train = zip(*tmp)

    return np.array(x_train), y_train
    
    
if __name__ == '__main__':
    name, x_t, y_t, size = EMNIST_data()
    print(type(x_t), type(x_t[0]))
    print(x_t[0:1].shape)
    print(y_t[0])
    x_t, y_t = data_shuffle(x_t, y_t)
    print(type(x_t), type(x_t[0]))
    print(x_t[0:1].shape)
    print(y_t[0])

