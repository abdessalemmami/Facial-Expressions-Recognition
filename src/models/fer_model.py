####
# Copyright (c) Abdessalem Mami <abdessalem.mami@esprit.tn>. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
####


import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,BatchNormalization, Activation,GlobalMaxPooling2D,Flatten, Dense, Dropout, Rescaling, RandomFlip, RandomContrast, RandomRotation, RandomZoom, Input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# CONV 64 X2 
# BATCH + MAX + DROPOUT
# CONV 128 X2
# BATCH + MAX + DROPOUT

def build_model(input_shape):
    num_classes = 7
    model = Sequential(
    [   #tf.keras.layers.RandomBrightness(factor=0.2),
        #tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.Rescaling(scale=1./127.5, offset=-1), 
    ]
    )
    reg = 0.001
	
    # CNN
    model.add(Conv2D(64, (5, 5), activation='relu', padding="same", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same", kernel_regularizer=tf.keras.regularizers.L2(reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3), activation='relu', padding="same", kernel_regularizer=tf.keras.regularizers.L2(reg)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same", kernel_regularizer=tf.keras.regularizers.L2(reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3), activation='relu', padding="same", kernel_regularizer=tf.keras.regularizers.L2(reg)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same", kernel_regularizer=tf.keras.regularizers.L2(reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))
    model.add(Flatten())


	# Fully Connected Layers
    model.add(Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg)))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg)))
    model.add(Dropout(0.2))
    
    # Output Layer
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.L2(reg)))

    model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=1e-3),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])
    return model

