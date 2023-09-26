import tensorflow as tf
import matplotlib.pyplot as plt


# load data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


# model build
# 1'st layer
inputs = tf.keras.layers.Input(shape = (32,32,3))
x = tf.keras.layers.Conv2D (filters = 64, kernel_size = 3, padding ='same', activation='relu')(inputs)
x = tf.keras.layers.Conv2D (filters = 64, kernel_size = 3, padding ='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
x = tf.keras.layers.Dropout(0.25)(x)

# hint : 3 more layers
###############################
###     Fill this layers    ###
###############################

# 5'th layer
x = tf.keras.layers.Conv2D (filters = 512, kernel_size = 3, padding ='same', activation='relu')(x)
x = tf.keras.layers.Conv2D (filters = 512, kernel_size = 3, padding ='same', activation='relu')(x)
x = tf.keras.layers.Conv2D (filters = 512, kernel_size = 3, padding ='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units = 4096, activation ='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(units = 512, activation ='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
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