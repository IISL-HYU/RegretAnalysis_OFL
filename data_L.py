
def CIFAR_10_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.reshape((50000, 32, 32, 3))
    x_train = x_train / 255.0
    
    return x_train, y_train, -2


if __name__ == '__main__':
    test = []
    for i in range(5):
        test.append(random.uniform(-10, 10))
    print(test)
    quant = quantizer(test, [True, 10, 10])
    print(quant)