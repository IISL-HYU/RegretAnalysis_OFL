import random
import numpy as np
import tensorflow as tf
import math

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



# def quantizer(grad, quantize):

#     s = quantize[1]
#     b = quantize[2]
    
#     q_grad = [(tf.Variable(grad[i])) for i in range(len(grad))]
#     grad_shape = [None for i in range(len(q_grad))]
#     for i in range(len(q_grad)):
#         grad_shape[i] = q_grad[i].numpy().shape
#     grad_flat = []
#     grad_tmp = []
#     for i in range(len(q_grad)):
#         grad_flat = np.append(grad_flat, q_grad[i])
#         grad_tmp.append(q_grad[i])
    
#     D = len(grad_flat)
#     print(D)
    
#     q_grad_flat = grad_flat

#     for i in range(D//b):
#         norm = np.linalg.norm(grad_tmp[i*b : (i+1)*b])
#         for j in range(i*b, (i+1)*b):
#             m_float = (np.abs(grad_tmp[j]) / norm) * s
#             m = int(m_float)
#             prob = m_float - m
#             xi_list = [m/s, (m+1)/s]
#             xi = np.random.choice(xi_list, 1, p=[1 - prob, prob])
#             if grad_flat[j] < 0 :
#                 xi[0] *= -1
#             q_grad_flat[j] = norm * xi[0]   
#     norm = np.linalg.norm(grad_flat[(D//b) * b : D])
#     for j in range((D//b) * b, D):
#         m_float = (np.abs(grad_flat[j]) / norm) * s
#         m = int(m_float)
#         prob = m_float - m
#         xi_list = [m/s, (m+1)/s]
#         xi = np.random.choice(xi_list, 1, p=[1 - prob, prob])
#         if grad_flat[j] < 0 :
#             xi[0] *= -1
#         q_grad_flat[j] = norm * xi[0]
    
#     q_grad_tmp = [None for i in range(len(grad_shape))]
#     bound_bef, bound_aft = 0, 0
#     for i in range(len(grad_shape)):
#         mulp = 1
#         for j in range(len(grad_shape[i])):
#             mulp = mulp * grad_shape[i][j]
#         bound_bef = bound_aft
#         bound_aft = bound_aft + mulp
#         q_grad_tmp[i] = q_grad_flat[bound_bef:bound_aft].reshape(grad_shape[i])
#     for i in range(len(grad)):
#         q_grad[i].assign(q_grad_tmp[i])
        
#     return q_grad


if __name__ == '__main__':
    test = []
    for i in range(5):
        test.append(random.uniform(-10, 10))
    print(test)
    quant = quantizer(test, [True, 10, 10])
    print(quant)