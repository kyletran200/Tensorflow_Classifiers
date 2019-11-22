from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


from tensorflow.keras import datasets, models, layers, backend

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from tensorflow.keras import Model

import matplotlib.pyplot as plt

import os


os.environ['KMP_DUPLICATE_LIB_OK']='True'


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

models = models.Sequential()





model.summary()

# Compile our model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=10, epochs=10, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print("This is test accuracy: " + str(test_acc))
