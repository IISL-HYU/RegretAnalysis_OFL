from data_L import CIFAR_10_data, MNIST_data, EMNIST_data, data_shuffle

data, x_train, y_train, input_size = CIFAR_10_data()


def test(data):
    print(type(data))
    
    
test(y_train[0:100])
