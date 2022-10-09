from http import client
import random
import tensorflow as tf
import keras
from keras import layers, models

class FL_Model(list):
    def __init__(self, K, quantize, prob, batch_size):
        super(FL_Model, self).__init__()
        
        self.K = K
        self.quantize = quantize
        self.prob = prob
        self.batch_size = batch_size
        
        for i in range(K):
            client_model = FedOGDModel(batch_size=batch_size)
            self.append(client_model)
        server_model = FedOGDModel(batch_size=batch_size)
        self.append(server_model)
    
    def train(self, x_train, y_train):
        K = self.K
        grad_list = []
        loss_avg, acc_avg = 0, 0
        is_first = True
        client_list = random_selection(K, self.prob)
        
        for i in range(K):
            result = self[i].train(x_train[i * self.batch_size : (i + 1) * self.batch_size], y_train[i * self.batch_size : (i + 1) * self.batch_size], self.quantize)
            grad_list.append(result[0])
            loss_avg += result[1]
            acc_avg += result[2]
        loss_avg = loss_avg / K
        acc_avg = acc_avg
        
        for i in client_list:
            if is_first:
                grad_avg = grad_list[i]
                is_first  = False
            else :
                for j in range(len(grad_avg)):
                    grad_avg[j] += grad_list[i][j]
                    
        for j in range(len(grad_avg)):
            grad_avg[j] /= len(client_list)
        self[K].optimizer.apply_gradients(zip(grad_avg, self[K].trainable_variables))
        
        for i in range(K):
            self[i].set_weights(self[K].get_weights())
        
        return loss_avg, acc_avg
    
    
class FedOGDModel(tf.keras.Model):
    def __init__(self, batch_size):
        super(FedOGDModel, self).__init__()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.batch_size = batch_size
        
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
        
       
    def train(self, x_train, y_train, quantize):
        with tf.GradientTape() as tape:
            y_pred = self(x_train, training = True)
            loss = self.loss(y_train, y_pred)
        gradient = tape.gradient(loss, self.trainable_variables)
        self.metric.update_state(y_train, y_pred)
        accuracy = self.metric.result().numpy()
        
        # if quantize:
        #     gradient = quantize(gradient)
        
        return gradient, loss.numpy(), accuracy
    
    def call(self, inputs):
        return self.dense(inputs)


def random_selection(K, prob):
    select_list = list(range(0, K))
    count = int(K * prob)

    return random.sample(select_list, count)