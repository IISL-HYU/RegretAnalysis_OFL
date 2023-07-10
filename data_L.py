import random
import tensorflow as tf

def CIFAR_10_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.reshape((50000, 32, 32, 3))
    x_train = x_train / 255.0
    
    return 'CIFAR_10', x_train, y_train, -2

def EMNIST_data():
    return 

def data_shuffle(x_train, y_train):
    tmp = [[x,y] for x, y in zip(x_train, y_train)]
    random.shuffle(tmp)
    x_train = [n[0] for n in tmp]
    y_train = [n[1] for n in tmp]
    
    return x_train, y_train
    
    
if __name__ == '__main__':
    x, y, size = CIFAR_10_data()
    print(x[0])
    print(y[0])