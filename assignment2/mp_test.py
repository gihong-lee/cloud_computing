import tensorflow as tf
import numpy as np

width = 32
height = 32
channels = 3
num_classes = 10 
batch_size = 512

activation = 'relu'
optimizer = 'adam'
# load data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)

y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

# model build
with tf.device("/device:GPU:0"):
    inputs = tf.keras.layers.Input(shape = (width,height,channels))
    x = tf.keras.layers.Conv2D(32, 3, activation=activation)(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation=activation)(x)
    x = tf.keras.layers.Flatten()(x)

with tf.device("/device:GPU:1"):
    x = tf.keras.layers.Dense(128, activation=activation)(x)
    x = tf.keras.layers.Dense(64, activation=activation)(x)
    output = tf.keras.layers.Dense(num_classes)(x)

model = tf.keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer = optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# model train
history = model.fit(x_train, y_train, epochs=50, batch_size=batch_size)
