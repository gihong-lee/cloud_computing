import tensorflow as tf
import matplotlib.pyplot as plt

def conv_block(input_tensor, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    shortcut = tf.keras.layers.Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def identity_block(input_tensor, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x


# load data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


# model build
inputs = tf.keras.layers.Input(shape = (32,32,3))
x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

# 1'st layer
x = conv_block(x, 64)
x = identity_block(x, 64)

# hint : 2 more layers
###############################
###     Fill this layers    ###
###############################

# 4'th layer
x = conv_block(x, 512)
x = identity_block(x, 512)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# model train
history = model.fit(x_train, y_train, epochs=50, batch_size=256, validation_data=(x_test, y_test))


# plot convergence graph
fig, ax = plt.subplots(figsize=(8, 5))

###############################
###  plt convergence graph  ###
###############################