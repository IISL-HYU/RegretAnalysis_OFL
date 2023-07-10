import math
import random
import numpy as np
import tensorflow as tf

def random_selection(K, prob):
    select_list = list(range(0, K))
    count = int(K * prob)

    return random.sample(select_list, count)

def quantize(g, s):
    select_list = [0, 1]
    quan_g = g.copy()
    g_abs = np.linalg.norm(g, 2)
    for i in range(len(g)):
        for l in range(s):
            if(g_abs * (l/s) <= abs(g[i]) and abs(g[i]) < g_abs * (l+1) / s):
                p = (abs(g[i]) / g_abs) * s - l
                distri = [1-p, p]
                l_temp = random.choices(select_list, distri)[0]
                quan_g[i] = (l + l_temp) / s * g_abs
                if(g[i] < 0):
                    quan_g[i] = -1 * quan_g[i]
                break
    return quan_g

def quantizer(grd_sum, quant):
    
    s = quant[1]
    b = quant[2]
    
    q_grd_sum = [(tf.Variable(grd_sum[i])) for i in range(len(grd_sum))]
    model_params = [None for i in range(len(q_grd_sum))]
    for i in range(len(q_grd_sum)):
        model_params[i] = q_grd_sum[i].numpy().shape
    all_params = []
    for i in range(len(q_grd_sum)):
        all_params = np.append(all_params, q_grd_sum[i])
    div_len = math.ceil(len(all_params) / b)  
    for i in range(b):
        temp_params = all_params[i*div_len:(i+1)*div_len]
        temp_params = quantize(temp_params, s)
        all_params[i*div_len:(i+1)*div_len] = temp_params
    q_grd_sum_list = [None for i in range(len(model_params))]
    bound_bef, bound_aft = 0, 0
    for i in range(len(model_params)):
        mulp = 1
        for j in range(len(model_params[i])):
            mulp = mulp * model_params[i][j]
        bound_bef = bound_aft
        bound_aft = bound_aft + mulp
        q_grd_sum_list[i] = all_params[bound_bef:bound_aft].reshape(model_params[i])
    for i in range(len(grd_sum)):
        q_grd_sum[i].assign(q_grd_sum_list[i])
    return q_grd_sum


def grad_norm_sq(grad):
    norm_sq = 0
    grad_flat = np.array([])
    for i in range(len(grad)):
        grad[i] = np.reshape(grad[i], -1)
        grad_flat = np.concatenate((grad_flat, grad[i]))
    norm_sq = np.dot(grad_flat, grad_flat)
    return norm_sq

def sigma_diff(model, x_train, y_train, iter):
    sigma_sq = 0
    K = model.K
    for i in range(iter):
        for j in range(K):
            with tf.GradientTape() as tape:
                y_pred = model[j](x_train[K*i+j:K*i+j+1], training = False)
                loss = model[j].loss(y_train[K*i+j], y_pred)
            gradient = tape.gradient(loss, model[j].trainable_variables)
            sigma_sq += grad_norm_sq(gradient) / K

    return sigma_sq / iter

if __name__ == '__main__':
    test = []
    for i in range(5):
        test.append(random.uniform(-10, 10))
    print(test)
    quant = quantizer(test, [True, 10, 10])
    print(quant)