import numpy as np
import tensorflow as tf
from keras import layers
from utils import random_selection, quantizer

class OFL_Model(list):
    def __init__(self, name, task, K, quantize, prob, L, input_size):
        super(OFL_Model, self).__init__()
        
        self.name = name
        self.K = K
        self.quantize = quantize
        self.prob = prob
        self.L = L
        self.grad_len = 0
        self.grad_info = ()
        self.result_list = []
        self.latest_result = 0
        self.input_size = input_size
        
        if input_size == -1 :
            for i in range(K):
                client_model = CNN_device()
                self.append(client_model)
            server_model = CNN_device()
            self.append(server_model)
        elif input_size == -2 :
            for i in range(K):
                client_model = CNN_2_device()
                self.append(client_model)
            server_model = CNN_2_device()
            self.append(server_model)
        elif input_size == -3 :
            for i in range(K):
                client_model = CNN_3_device()
                self.append(client_model)
            server_model = CNN_3_device()
            self.append(server_model)
        elif task == 'clf':
            for i in range(K):
                client_model = Clf_device(input_size)
                self.append(client_model)
            server_model = Clf_device(input_size)
            self.append(server_model)
        elif task == 'reg':
            for i in range(K):
                client_model = Reg_device(input_size)
                self.append(client_model)
            server_model = Reg_device(input_size)
            self.append(server_model)
        elif task == 'time':
            for i in range(K):
                client_model = Time_device(window=input_size)
                self.append(client_model)
            server_model = Time_device(window=input_size)
            self.append(server_model)
    
    def pre_train(self, x_train, y_train):
        K = self.K
        self[K].fit(x_train, y_train, epochs=1)
        weights = self[K].get_weights()
        
        return weights
        
            
    def train(self, x_train, y_train, is_period):
        K = self.K
        client_list = random_selection(K, self.prob)
        
        #Local Training
        result = 0
        for i in range(K):
            result += self[i].train(x_train[i:i+1], y_train[i:i+1], is_period, self.L)
        if not is_period:
            for i in range(self.L):
                self.latest_result += result
                self.result_list.append(self.latest_result)

            #Transmission
            grad_sample = self[0].gradient_sum
            if self.grad_len == 0:
              self.grad_len = len(grad_sample)
              for i in range(self.grad_len):
                  layer_shape = grad_sample[i].get_shape()
                  layer_len = len(grad_sample[i].numpy().flatten())
                  self.grad_info += (layer_shape, layer_len)
            
            grad = []
            grad_len = self.grad_len
            grad_info = self.grad_info
            for idx, i in enumerate(client_list):
                grad_flat = np.array([])
                grad_tmp = self[i].gradient_sum
                for j in range(grad_len):
                    tmp_flat = grad_tmp[j].numpy().flatten()
                    grad_flat = np.hstack((grad_flat, tmp_flat))
                grad_flat /= self.prob
                if self.quantize[0]:
                    grad_flat = quantizer(grad_flat, self.quantize)
                if idx == 0 : grad = grad_flat
                else: grad += grad_flat

            grad /= K
            grad_final = []
            index = 0
            for i in range(grad_len):
                arr = grad[index:index+grad_info[2*i + 1]]
                index += grad_info[2*i + 1]
                grad_final.append(tf.convert_to_tensor(arr.reshape(grad_info[2*i]), dtype="float32"))
            self[K].optimizer.apply_gradients(zip(grad_final, self[K].trainable_variables))            

            weights = self[K].get_weights()
            for i in range(K):
                self[i].gradient_sum = 0
                self[i].set_weights(weights)
    
    def pull_result(self):
        K = self.K
        result = []
        for i in range(len(self.result_list)):
            result.append(self.result_list[i] / (K * (i+1)))
        return result
    
    def pull_last_result(self):
        K = self.K
        last_result = self.result_list[len(self.result_list) - 1] / (K * (len(self.result_list)))
        return last_result
    
    
class CNN_device(tf.keras.Model):
    def __init__(self):
        super(CNN_device, self).__init__()

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
        
       
    def train(self, x_train, y_train, is_period, L):
        with tf.GradientTape() as tape:
            y_pred = self(x_train, training = True)
            loss = self.loss(y_train, y_pred)
        gradient = tape.gradient(loss, self.trainable_variables)
        self.metric.update_state(y_train, y_pred)
        accuracy = self.metric.result().numpy()
        
        if L == 1 or is_period == 1:
            self.gradient_sum = gradient
        else :
            for i in range(len(gradient)):
                self.gradient_sum[i] += gradient[i]
        print(accuracy)
        return accuracy
    
    def call(self, inputs):
        return self.dense(inputs)

class CNN_2_device(tf.keras.Model):
    def __init__(self):
        super(CNN_2_device, self).__init__()

        self.gradient_sum = 0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy()
        
        self.dense = tf.keras.Sequential([
            tf.keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ])
        tf.random.set_seed(3)
        self.compile(optimizer = self.optimizer, loss = self.loss)
        
    def train(self, x_train, y_train, is_period, L):
        with tf.GradientTape() as tape:
            y_pred = self(x_train, training = True)
            loss = self.loss(y_train, y_pred)
        gradient = tape.gradient(loss, self.trainable_variables)
        self.metric.update_state(y_train, y_pred)
        accuracy = self.metric.result().numpy()
        
        if L == 1 or is_period == 1:
            self.gradient_sum = gradient
        else :
            for i in range(len(gradient)):
                self.gradient_sum[i] += gradient[i]
        return accuracy
    
    def call(self, inputs):
        return self.dense(inputs)

class CNN_3_device(tf.keras.Model):
    def __init__(self):
        super(CNN_3_device, self).__init__()

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
            layers.Dense(26, activation="softmax"),
        ])
        tf.random.set_seed(3)
        self.compile(optimizer = self.optimizer, loss = self.loss)
        
       
    def train(self, x_train, y_train, is_period, L):
        with tf.GradientTape() as tape:
            y_pred = self(x_train, training = True)
            loss = self.loss(y_train, y_pred)
        gradient = tape.gradient(loss, self.trainable_variables)
        self.metric.update_state(y_train, y_pred)
        accuracy = self.metric.result().numpy()
        
        if L == 1 or is_period == 1:
            self.gradient_sum = gradient
        else :
            for i in range(len(gradient)):
                self.gradient_sum[i] += gradient[i]
        return accuracy
    
    def call(self, inputs):
        return self.dense(inputs)

class Clf_device(tf.keras.Model):
    def __init__(self, input_size):
        super(Clf_device, self).__init__()

        self.gradient_sum = 0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.dense = tf.keras.Sequential([
            tf.keras.Input(shape=(None, 1, input_size)),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(4, activation="softmax"),
        ])
        tf.random.set_seed(3)
        self.compile(optimizer = self.optimizer, loss = self.loss)
       
    def train(self, x_train, y_train, is_period, L):
        with tf.GradientTape() as tape:
            y_pred = self(x_train, training = True)
            loss = self.loss(y_train, y_pred)
        gradient = tape.gradient(loss, self.trainable_variables)
        self.metric.update_state(y_train, y_pred)
        accuracy = self.metric.result().numpy()
        
        if L == 1 or is_period == 1:
            self.gradient_sum = gradient
        else :
            for i in range(len(gradient)):
                self.gradient_sum[i] += gradient[i]
        return accuracy
    
    def call(self, inputs):
        return self.dense(inputs)

class Reg_device(tf.keras.Model):
    def __init__(self, input_size):
        super(Reg_device, self).__init__()
        
        self.gradient_sum = 0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
        self.bias_initializer=tf.keras.initializers.Zeros()
        self.input_size = input_size
        
        self.dense = tf.keras.Sequential([
            tf.keras.Input(shape=(input_size, 1)),
            layers.Dense(64, activation='relu', kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer),
            layers.Dense(64, activation='relu', kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer),
            layers.Dense(1)
        ])
        tf.random.set_seed(3)
        self.compile(optimizer = self.optimizer, loss = self.loss)
        
    def train(self, x_train, y_train, is_period, L):
        with tf.GradientTape() as tape:
            y_pred = self(x_train, training = True)
            loss = self.loss(y_train, y_pred)
        gradient = tape.gradient(loss, self.trainable_variables)
        
        if L == 1 or is_period == 1:
            self.gradient_sum = gradient
        else :
            for i in range(len(gradient)):
                self.gradient_sum[i] += gradient[i]
        
        return loss.numpy()
    
    def call(self, inputs):
        return self.dense(inputs)

class Time_device(tf.keras.Model):
    def __init__(self, window):
        super(Time_device, self).__init__()
        
        self.gradient_sum = 0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss = tf.keras.losses.MeanSquaredError()
        # self.kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
        # self.bias_initializer=tf.keras.initializers.Zeros()
        self.window = window
        
        #MNIST CNN Model
        self.dense = tf.keras.Sequential([
            tf.keras.Input(shape=(1, window)),
            layers.Dense(16, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        tf.random.set_seed(3)
        self.compile(optimizer = self.optimizer, loss = self.loss)

    def train(self, x_train, y_train, is_period, L):
        with tf.GradientTape() as tape:
            y_pred = self(x_train, training = True)
            loss = self.loss(y_train, y_pred)
        gradient = tape.gradient(loss, self.trainable_variables)
        
        if L == 1 or is_period == 1:
            self.gradient_sum = gradient
        else :
            for i in range(len(gradient)):
                self.gradient_sum[i] += gradient[i]
                
        return loss.numpy() 
    
    def call(self, inputs):
        return self.dense(inputs)
    

if __name__ == '__main__':
  
  import tensorflow as tf
  import time

  model = OFL_Model('OFedIQ', 'clf', 5, [True, 17, 1134], 0.5151, 1, -1)
  #OFL_Model('FedOGD', 'clf', 5, [False, 0, 0], 1, 1, -1)

  (x,y), (x1,y1) = tf.keras.datasets.mnist.load_data()
  x = x.reshape((60000, 28, 28, 1))
  x = x / 255.0
  print(time.ctime())
  for j in range(100):
    for i in range(1):
      model.train(x[5*i : 5*(i+1)], y[5*i : 5*(i+1)], 0)
    print(model.pull_last_result())

