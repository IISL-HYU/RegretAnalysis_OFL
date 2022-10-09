import tensorflow as tf
import keras
from keras import layers, models
from utils import random_selection, quantizer

class OFL_Model(list):
    def __init__(self, name, task, K, quantize, prob, L):
        super(OFL_Model, self).__init__()
        
        self.name = name
        self.K = K
        self.quantize = quantize
        self.prob = prob
        self.L = L
        self.result_list = []
        
        if task == 'clf':
            for i in range(K):
                client_model = Clf_device()
                self.append(client_model)
            server_model = Clf_device()
            self.append(server_model)
        # elif task == 'reg':
        #     for i in range(K):
        #         client_model = Reg_device(L=L)
        #         self.append(client_model)
        #     server_model = Reg_device(L=L)
        #     self.append(server_model)
    
    def train(self, x_train, y_train, is_period):
        K = self.K
        grad_list = []
        result = 0
        client_list = random_selection(K, self.prob)
        
        #Local Training
        for i in range(K):
            result += self[i].train(x_train[i:i+1], y_train[i:i+1], is_period)
        self.result_list.append(result)
        
        #Transmission
        if not is_period:
            for i in client_list:
                grad_list.append(self[i].gradient_sum)
            
            if self.quantize[0]:
                grad_avg = quantizer(grad_list, self.quantize)
            else:
                grad_avg = grad_list[0]
                for i in range(1, len(client_list)):
                    for j in range(len(grad_avg)):
                        grad_avg[j] += grad_list[i][j]

            for j in range(len(grad_avg)):
                grad_avg[j] /= len(client_list)
            self[K].optimizer.apply_gradients(zip(grad_avg, self[K].trainable_variables))            

            for i in range(K):
                self[i].gradient_sum = 0
                self[i].set_weights(self[K].get_weights())
    
    def pull_result(self):
        K = self.K
        result = []
        for i in range(len(self.result_list)):
            result.append(self.result_list[i] / (K * (i+1)))
        return result
    
    
class Clf_device(tf.keras.Model):
    def __init__(self):
        super(Clf_device, self).__init__()
        
        self.gradient_sum = 0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy()
        
        #MNIST CNN Model
        self.dense = tf.keras.Sequential([
            tf.keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation="softmax"),
        ])
        tf.random.set_seed(3)
        self.compile(optimizer = self.optimizer, loss = self.loss)
        
       
    def train(self, x_train, y_train, is_period):
        with tf.GradientTape() as tape:
            y_pred = self(x_train, training = True)
            loss = self.loss(y_train, y_pred)
        gradient = tape.gradient(loss, self.trainable_variables)
        self.metric.update_state(y_train, y_pred)
        accuracy = self.metric.result().numpy()
        
        if is_period == 1:
            self.gradient_sum = gradient
        else :
            for i in range(len(gradient)):
                self.gradient_sum[i] += gradient[i]
                
        return accuracy
    
    def call(self, inputs):
        return self.dense(inputs)


# class Reg_device(tf.keras.Model):
#     def __init__(self, L):
#         super(Clf_device, self).__init__()
        
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
#         self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
#         self.L = L
        
#         #MNIST CNN Model
#         self.dense = tf.keras.Sequential([
#             tf.keras.Input(shape=(28, 28, 1)),
#             layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Flatten(),
#             layers.Dense(10, activation="softmax"),
#         ])
#         tf.random.set_seed(3)
#         self.compile(optimizer = self.optimizer, loss = self.loss)
        
       
#     def train(self, x_train, y_train, quantize):
#         with tf.GradientTape() as tape:
#             y_pred = self(x_train, training = True)
#             loss = self.loss(y_train, y_pred)
#         gradient = tape.gradient(loss, self.trainable_variables)
        
#         if quantize:
#             gradient = quantize(gradient)
        
#         return gradient, loss.numpy()
    
#     def call(self, inputs):
#         return self.dense(inputs)