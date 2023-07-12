# import tensorflow as tf
# import numpy as np

# info = ([3, 3, 1, 32], 288, [32], 32, [3, 3, 32, 64], 18432, [64], 64, [1600, 10], 16000, [10], 10)
# b = np.array(list(range(34826)))
# print(b)

# index = 0
# final=[]
# for i in range(6):
#     arr = b[index:index+info[2*i + 1]]
#     index += info[2*i + 1]
#     final.append(tf.convert_to_tensor(arr.reshape(info[2*i]), dtype="float32"))
# print(final)

# grad_info = ()
# final_tmp = final
# grad_len = len(final_tmp)
# for i in range(grad_len):
#     layer_shape = final_tmp[i].get_shape()
#     layer_len = len(final_tmp[i].numpy().flatten())
#     grad_info += (layer_shape, layer_len)

# grad_flat = np.array([])
# for j in range(grad_len):
#     tmp_flat = final[j].numpy().flatten()
#     grad_flat = np.hstack((grad_flat, tmp_flat))

# print(grad_flat)
# grad = []
# grad = grad_flat
# grad += grad_flat
# print(grad)

# grad /= 2
# print(grad)

# ffinal = []
# index = 0
# for i in range(6):
#     arr = grad[index:index+info[2*i + 1]]
#     index += info[2*i + 1]
#     ffinal.append(tf.convert_to_tensor(arr.reshape(info[2*i]), dtype="float32"))
# print(ffinal)


print(1.213123//1)