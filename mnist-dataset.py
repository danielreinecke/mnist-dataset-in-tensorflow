import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
from keras import layers, datasets

keras.datasets.mnist.load_data(path="mnist.npz")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = tf.cast(tf.reshape(x_train, (-1, 28*28)), tf.float32) / 255.0
x_test = tf.cast(tf.reshape(x_test, (-1, 28*28)), tf.float32) / 255.0



model = keras.Sequential(
    [
        layers.Dense(512, activation = 'relu'),
        layers.Dense(256, activation = 'relu'),
        layers.Dense(32, activation = 'relu'),
        layers.Dense(10)
    ]
)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
model.evaluate(x_train, y_train, batch_size=32, verbose=2)