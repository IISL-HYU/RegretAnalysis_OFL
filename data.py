from keras import datasets

def MNIST_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train / 255.0

    return x_train, y_train


def Air_data():
    return

def Syn_data():
    return