import tensorflow as tf
import numpy as np
from keras_flops import get_flops
import time

width = 32
height = 32
channels = 3
num_classes = 10 
batch_size = 512

activation = 'relu'
optimizer = 'adam'

def res50():
    def conv_block(input_tensor, num_filters):
        x = tf.keras.layers.Conv2D(num_filters, (1, 1), strides=1, padding='valid')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)

        x = tf.keras.layers.Conv2D(num_filters, (3, 3), strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)

        x = tf.keras.layers.Conv2D(num_filters*4, (1, 1), strides=1, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        shortcut = tf.keras.layers.Conv2D(num_filters*4, (1, 1), strides=1, padding='valid')(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.Activation(activation)(x)
        return x

    def identity_block(input_tensor, num_filters):
        x = tf.keras.layers.Conv2D(num_filters, (1, 1), strides=1, padding='valid')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)

        x = tf.keras.layers.Conv2D(num_filters, (3, 3), strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)

        x = tf.keras.layers.Conv2D(num_filters*4, (1, 1), strides=1, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.add([x, input_tensor])
        x = tf.keras.layers.Activation(activation)(x)
        return x

    # model build
    inputs = tf.keras.layers.Input(shape = (width,height,channels))
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(inputs)
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(x)

    # 1'st layer
    x = conv_block(x, 64)
    x = identity_block(x, 64)
    x = identity_block(x, 64)

    # 2'nd layer
    x = conv_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)

    x = conv_block(x, 256)
    for _ in range(5):
        x = identity_block(x, 256)

    # 4'th layer
    x = conv_block(x, 512)
    x = identity_block(x, 512)
    x = identity_block(x, 512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer = optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def vgg16():
    inputs = tf.keras.layers.Input(shape = (width,height,channels))

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', activation=activation)(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', activation=activation)(x)

    x = tf.keras.layers.MaxPool2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding='same', activation=activation)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding='same', activation=activation)(x)

    x = tf.keras.layers.MaxPool2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding='same', activation=activation)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding='same', activation=activation)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding='same', activation=activation)(x)

    x = tf.keras.layers.MaxPool2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, padding='same', activation=activation)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, padding='same', activation=activation)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, padding='same', activation=activation)(x)

    x = tf.keras.layers.MaxPool2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, padding='same', activation=activation)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, padding='same', activation=activation)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=1, padding='same', activation=activation)(x)

    x = tf.keras.layers.MaxPool2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(4096, activation=activation)(x)

    x = tf.keras.layers.Dense(512, activation=activation)(x)

    x = tf.keras.layers.Dense(512, activation=activation)(x)

    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer = optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def simple_model():
    inputs = tf.keras.layers.Input(shape = (width,height,channels))
    x = tf.keras.layers.Conv2D(32, 3, activation=activation)(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation=activation)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation=activation)(x)
    x = tf.keras.layers.Dense(64, activation=activation)(x)
    output = tf.keras.layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer = optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

model = simple_model()

# model profile
flops = get_flops(model, batch_size=batch_size)
print(f"FLOPS: {flops / 10 ** 9:.03} B")

model.summary()
