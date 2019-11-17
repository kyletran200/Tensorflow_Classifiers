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

# Resize images?
#trains_images = backend.resize_images(train_images, (224, 224))

print("Train: X=%s, y=%s" % (train_images.shape, train_labels.shape))   
print("Test: X=%s, y=%s" % (test_images.shape, test_labels.shape))


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
"""

model = models.Sequential()

# 2 Convolutional Layers
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

# 1 pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# 2 Convolutional Layers
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

# 1 Pooling Layer
model.add(layers.MaxPooling2D((2, 2)))

# 2 Convolutional Layers
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

# 1 Pooling Layer
model.add(layers.MaxPooling2D(2,2))

# 4 Convolutional Layers
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))

# 1 Pooling Layer
model.add(layers.MaxPooling2D(2,2))

# 4 Convolutional Layers
model.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu',padding='same'))

# 1 Pooling Layer
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# Compile our model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print("This is test accuracy: " + str(test_acc))

"""
class vgg19(Model):
    def __init__(self):
        super(vgg19, self).__init__()
        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.pool = MaxPooling2D(2,2)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv2(x)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

model = vgg19()
"""
