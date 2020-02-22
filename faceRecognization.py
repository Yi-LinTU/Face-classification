# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 16:30:04 2018

@author: yilin
"""
from __future__ import print_function
from skimage import io,transform
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras import layers
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.core import Dropout
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import cv2
import warnings
import os, sys
import numpy as np
import keras
import random
import keras.backend.tensorflow_backend as KTF
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)
KTF.set_session(session )

def dataset():
    
    #get dataset root directory
    dataset_rootdir = 'D:/Code/CVR/CroppedYale'
    sort_root = os.listdir(dataset_rootdir)
    sort_root.sort()
    #print(sort_root)
    
    directories = []
    x_train = []
    x_test  = []
    y_train = []
    y_test  = []

    label_ctr = 1
    train_num = 35
    
    
    for filename in sort_root:
        path = os.path.join(dataset_rootdir, filename)
        directories.append(path)
    
    
        
    #choose training data
    for i in directories:
        ctr = 0
        for filename in os.listdir(i):
            image_path = os.path.join(i, filename)
            image = cv2.imread(image_path)
           #print(image_path)
            
            if ctr < train_num:
                x_train.append(image)
                y_train.append(label_ctr)
            else:
                x_test.append(image)
                y_test.append(label_ctr)
            ctr = ctr + 1
            
        label_ctr = label_ctr + 1
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_test, y_test)
        
    
def Model_Training():
    class_num = 38
    epochs = 50
    
    img_rows, img_cols = 192, 168
    (x_train, y_train), (x_test, y_test) = dataset()
    
   
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    y_train = y_train - 1 
    y_train = keras.utils.to_categorical(y_train, class_num)
    y_test = y_test - 1
    y_test = keras.utils.to_categorical(y_test, class_num)
    #print(x_train)
    #print(y_train)
    #print(x_test)
    #print(y_test)
    
    print("x_train: ",x_train.shape) 
    print("x_test: ",x_test.shape)
    print("y_train: ",y_train.shape) 
    print("y_test: ",y_test.shape)
    
    #model
    model = Sequential()

    
    #model.add(Conv2D(8, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3, 3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(Conv2D(256, kernel_size=(3, 3),activation='relu',padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(Conv2D(512, kernel_size=(3, 3),activation='relu',padding='same'))
    #model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
    #model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(0.25))
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))

    #sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=adam,
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              epochs = epochs,
              verbose = 1,
              batch_size = 8,
              shuffle = 'True',
              validation_data = (x_test, y_test))
    
    model.save('my_model.h5')
    
    #model
        
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model
    
def main():
    Model_Training()
    
if __name__ == "__main__":
    main()
    
    
