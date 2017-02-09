# coding: utf-8
from __future__ import print_function

"""
VGG 16 model
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

def VGG_16(weights_path=None, dim_ordering='th'):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering, input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=dim_ordering))

    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(128, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1,1), dim_ordering=dim_ordering))
    model.add(Convolution2D(128, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=dim_ordering))

    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(256, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=dim_ordering))

    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=dim_ordering))

    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(ZeroPadding2D((1, 1), dim_ordering=dim_ordering))
    model.add(Convolution2D(512, 3, 3, activation='relu', dim_ordering=dim_ordering))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=dim_ordering))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    return model
