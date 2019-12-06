from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


from tensorflow.keras import datasets, models, layers, backend, Input

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation, Dropout, add

from tensorflow.keras import Model

import matplotlib.pyplot as plt

import os


os.environ['KMP_DUPLICATE_LIB_OK']='True'

stride = 1
CHANNEL_AXIS = 3

def res_layer(x ,filters,pooling = False,dropout = 0.0):
    temp = x
    temp = Conv2D(filters,(3,3),strides = stride,padding = "same")(temp)
    temp = BatchNormalization(axis = CHANNEL_AXIS)(temp)
    temp = Activation("relu")(temp)
    temp = Conv2D(filters,(3,3),strides = stride,padding = "same")(temp)

    x = add([temp,Conv2D(filters,(3,3),strides = stride,padding = "same")(x)])
    if pooling:
        x = MaxPooling2D((2,2))(x)
    if dropout != 0.0:
        x = Dropout(dropout)(x)
    x = BatchNormalization(axis = CHANNEL_AXIS)(x)
    x = Activation("relu")(x)
    return x

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

inp = Input(shape = (32,32,3))

x = inp
x = Conv2D(16,(3,3),strides = stride,padding = "same")(x)
x = BatchNormalization(axis = CHANNEL_AXIS)(x)
x = Activation("relu")(x)

x = res_layer(x,32,dropout = 0.0)
x = res_layer(x,32,dropout = 0.0)
x = res_layer(x,32,dropout = 0.1,pooling = True)
x = res_layer(x,64,dropout = 0.1)
x = res_layer(x,64,dropout = 0.2,pooling = True)
x = res_layer(x,256,dropout = 0.2)

x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(4096,activation = "relu")(x)
x = Dropout(0.23)(x)
x = Dense(100,activation = "softmax")(x)

model = Model(inp,x,name = "Resnet")

model.summary()

# Compile our model
model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=10, epochs=20, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print("This is test accuracy: " + str(test_acc))
