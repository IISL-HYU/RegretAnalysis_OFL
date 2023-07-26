import time
import pickle
import numpy as np

from data   import Room_data
from data_L import CIFAR_10_data, MNIST_data, EMNIST_data, data_shuffle
from model  import OFL_Model

K = 1000
D = 60442                                       # MNIST 34826 / CIFAR_10 150826(60362) / EMNIST 60442
P = 0.04

def opt_param(p1, D, print_result):
    
    min_index = 1
    min_bound = -1
    bound_list = []

    for s in range(1, 100):
        bound = np.log2(s + 1)/16 + 4 * np.power(p1/s, 2/3)
        bound_list.append(bound)
        alpha = np.power(p1/s, 2/3)
        p2 = 32 * p1/(32 * alpha + 1 + np.log2(s + 1))
        if (min_bound < 0 or min_bound > bound) and p2 < 1:
            min_bound = bound
            min_index = s
            min_alpha = alpha
            min_p2 = p2

    if print_result:
        print()
        print("- OFedIQ optimal parameter same communication overhead with OFedAvg (p = %.3f)" %(p1))
        print("- s =", min_index, ", alpha =", min_alpha)
        print("- b =", int(min_alpha * D), ", p =", min_p2)
        print()
    
    return min_index, min_alpha, int(min_alpha * D), min_p2
s, _, b, p = opt_param(P, D, True)

data, x_train, y_train, input_size = EMNIST_data()                                                # MNIST_data() #Room_data()
task = 'clf'                                                                                        # task type

Model_list = []
Model_list.append(OFL_Model('FedOGD', task, K, [False, 0, 0], 1, 1, input_size))
Model_list.append(OFL_Model('OFedAvg', task, K, [False, 0, 0], P, 1, input_size))
Model_list.append(OFL_Model('FedOMD', task, K, [False, 0, 0], 1, int(1/P), input_size))
Model_list.append(OFL_Model('OFedIQ', task, K, [True, s, b], p, 1, input_size))
print("========= Model_list is generated ===================")
print()

initial_weights = Model_list[0].pre_train(x_train[0:10], y_train[0:10], 1)                    # Pre-train data length
for model in Model_list:                                                                            ## C:20000, E:100, M:1
    for i in range(K+1):
        model[i].set_weights(initial_weights)

iter_max = 20                                                                                       # iter_max
i_max = len(y_train) // K                                                                           ## C:20, E:15, M:15
print()
print("Total timesteps :", iter_max*i_max, "| Data reuse :", iter_max, "| steps per dataset :", i_max)
print()

for iter in range(iter_max):
    print("========== iter", iter, "started ==================")
    x_train, y_train = data_shuffle(x_train, y_train)
    for model in Model_list:
        print("<", model.name, ">")
        for i in range(i_max):
            model.train(x_train[K*i : K*(i+1)], y_train[K*i : K*(i+1)], ((i_max * iter) + (i+1)) % model.L)
        last_acc = model.pull_last_result()
        print(model.name, "| Accuracy =", last_acc, "% | Time =", time.ctime())

for model in Model_list:
    result = model.pull_result()
    with open(f"./result_L/{task}_{model.name}_{data}_{K}_{P}_2.pkl","wb") as f:
        pickle.dump(result, f)
      
print("Finished.")