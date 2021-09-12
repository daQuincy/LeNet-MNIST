# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:47:31 2021

@author: YQ
"""
import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import mnist


def lenet(width, height, depth, classes):
    inputShape = (height, width, depth)
    model = Sequential()
    
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    
    return model
    

EPOCHS = 3
LR = 1e-3
BS = 64

(Xtr, ytr), (Xte, yte) = mnist.load_data()

Xtr = np.expand_dims(Xtr, -1)
Xte = np.expand_dims(Xte, -1)

Xtr = (Xtr / 255.0).astype(np.float32)
Xte = (Xte / 255.0).astype(np.float32)

ytr, yte = to_categorical(ytr, 10), to_categorical(yte, 10)

model = lenet(28, 28, 1, 10)
opt = Adam(lr=LR, decay=LR / EPOCHS)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, fill_mode="nearest")

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit_generator(aug.flow(Xtr, ytr, batch_size=BS),
	validation_data=(Xte, yte), steps_per_epoch=len(Xtr) // BS,
	epochs=EPOCHS, verbose=1)


model.save("save/digits.h5")