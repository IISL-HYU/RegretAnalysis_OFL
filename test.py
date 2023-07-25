import tensorflow as tf
import keras
from keras import layers

model = tf.keras.Sequential([
            tf.keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation="softmax"),
        ])
model.summary()