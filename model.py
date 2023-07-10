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
    
    def train(self, x_train, y_train, is_period):
        K = self.K
        grad_list = []
        q_grad_list = []
        client_list = random_selection(K, self.prob)
        
        #Local Training
        for i in range(K):
            self.latest_result += self[i].train(x_train[i:i+1], y_train[i:i+1], is_period, self.L)
        self.result_list.append(self.latest_result)
        
        #Transmission
        if not is_period:
            for i in client_list:
                grad_sum = self[i].gradient_sum
                for j in range(len(grad_sum)):
                    grad_sum[j] = grad_sum[j] / self.prob
                grad_list.append(grad_sum)
            
            if self.quantize[0]:
                for i in range(len(grad_list)):
                    q_grad_list.append(quantizer(grad_list[i], self.quantize))
            else:
                q_grad_list = grad_list
            grad_avg = q_grad_list[0]
            for i in range(1, len(client_list)):
                for j in range(len(grad_avg)):
                    grad_avg[j] = grad_avg[j] + q_grad_list[i][j]

            for j in range(len(grad_avg)):
                grad_avg[j] /= K
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
    
    def pull_last_result(self):
        K = self.K
        last_result = self.result_list[-1] / (K * (len(self.result_list)))
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
            layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
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