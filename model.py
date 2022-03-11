from cProfile import label
from astroNN.datasets import load_galaxy10
from tensorflow.keras import utils
import tensorflow as tf
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
import h5py

from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import matplotlib.pyplot as plt


with h5py.File('Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

# To convert the labels to categorical 10 classes
labels = utils.to_categorical(labels, 10)

# To convert to desirable type
labels = labels.astype(np.float32)
images = images.astype(np.float32)

print(len(images[0]))

# # To convert to desirable type
labels = labels.astype(np.float32)
images = images.astype(np.float32)


train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]



vgg16_model = tf.keras.applications.vgg16.VGG16()
model = Sequential()

for layer in vgg16_model.layers[:-1]:
  model.add(layer)
for layer in model.layers:
  layer.trainable = False
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x = images[train_idx], y = labels[train_idx], epochs = 1)