import math
import random
import numpy as np
import tensorflow as tf

def random_selection(K, prob):
    select_list = list(range(0, K))
    count = int(K * prob)

    return random.sample(select_list, count)


def quantizer(grad, quant):
    s = quant[1]
    b = quant[2]
    D = len(grad)
    part_size = D//b
    norm_list = np.zeros(b)

    for l in range(b-1):
        norm_list[l] = np.linalg.norm(grad[part_size*l : part_size*(l+1)], 2)
    norm_list[b-1] = np.linalg.norm(grad[part_size*(b-1) : D], 2)

    q_grad = np.zeros(D)
    for i in range(D):
        if i//part_size >= b : norm = norm_list[-1]
        else : norm = norm_list[i//part_size]
        if grad[i] == 0 : q_grad[i] = 0
        else: 
            sign = 1
            if grad[i] < 0 : sign = -1
            tmp = s * (abs(grad[i])/norm)
            m = tmp//1
            dist = [1-(tmp-m), tmp-m]
            xi = (m + random.choices([0,1], dist)[0]) / s
            q_grad[i] = norm * sign * xi
    return q_grad


# def sigma_diff(model, x_train, y_train, iter):
#     sigma_sq = 0
#     K = model.K
#     for i in range(iter):
#         for j in range(K):
#             with tf.GradientTape() as tape:
#                 y_pred = model[j](x_train[K*i+j:K*i+j+1], training = False)
#                 loss = model[j].loss(y_train[K*i+j], y_pred)
#             gradient = tape.gradient(loss, model[j].trainable_variables)
#             sigma_sq += grad_norm_sq(gradient) / K

#     return sigma_sq / iter

if __name__ == '__main__':
    test = []
    for i in range(5):
        test.append(random.uniform(-10, 10))
    print(test)
    quant = quantizer(test, [True, 10, 10])
    print(quant)