import random
import numpy as np
import tensorflow as tf

def random_selection(K, prob):
    select_list = list(range(0, K))
    count = int(K * prob)

    return random.sample(select_list, count)

def quantizer(grad, quantize):

    s = quantize[1]
    b = quantize[2]
    p = quantize[3]
    D = len(grad)
    q_grad = []

    for i in range(D//b):
        norm = np.linarg.norm(grad[i*b : (i+1)*b - 1])
        for j in range(i*b, (i+1)*b):
            tmp

            if grad[j] < 0 :
                tmp *= -1
            q_grad.append(tmp)

    # if D%b :
    #     grad[i*b : 

    # q_grd_sum = [(tf.Variable(grd_sum[i])) for i in range(len(grd_sum))]
    # model_params = [None for i in range(len(q_grd_sum))]
    # for i in range(len(q_grd_sum)):
    #     model_params[i] = q_grd_sum[i].numpy().shape
    # all_params = []
    # for i in range(len(q_grd_sum)):
    #     all_params = np.append(all_params, q_grd_sum[i] / p) # divided by probability (OFedQIT)
    # div_len = math.ceil(len(all_params) / b)  
    # for i in range(b):
    #     temp_params = all_params[i*div_len:(i+1)*div_len]
    #     temp_params = quantize(temp_params, s)
    #     all_params[i*div_len:(i+1)*div_len] = temp_params
    # q_grd_sum_list = [None for i in range(len(model_params))]
    # bound_bef, bound_aft = 0, 0
    # for i in range(len(model_params)):
    #     mulp = 1
    #     for j in range(len(model_params[i])):
    #         mulp = mulp * model_params[i][j]
    #         bound_bef = bound_aft
    #         bound_aft = bound_aft + mulp
    #         q_grd_sum_list[i] = all_params[bound_bef:bound_aft].reshape(model_params[i])
    # for i in range(len(grd_sum)):
    #     # Time check
    #     # work_start = int(time.time() * 1000.0)
    #     q_grd_sum[i].assign(q_grd_sum_list[i])
    # # # Return a dict mapping metric names to current value
    # return q_grd_sum