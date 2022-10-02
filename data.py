from keras import datasets

def pull_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_test = x_test[0:100]
    y_test = y_test[0:100]

    return x_train, y_train, x_test, y_test