import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, model_from_json, Model 
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random
import cv2
import dlib
import pandas as pd
import importlib.util
import sys, os
from pathlib import Path
#import core

ROOT_DIR = Path(__file__).parents[1]
sys.path.append(os.path.abspath(ROOT_DIR))
from utils.load_dataset import load_dataset_with_facial_features


class Combined_Facial_Net:
    def __init__(self, input_img_shape = None, 
                       input_features_shape = None, 
                       num_classes = None,
                       json_filename=None, h5_filename=None):
        self.__model = None
        if json_filename is not None:
            self.load_model(json_filename)
            if h5_filename is not None:
                self.load_weights(h5_filename)
        elif input_img_shape and input_features_shape and num_classes:
            #self.__model = self._features_model(input_features_shape, num_classes)
            self._init_model_combined_(input_img_shape, 
                              input_features_shape, 
                              num_classes)

    def _img_model_(self, input_img_shape, output_filters):
        inputs_img = Input(shape=input_img_shape, name='img_inputs')
        #* layer 1
        conv_1 = Conv2D(32, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(inputs_img)
        pool_1 = MaxPool2D()(conv_1)
        pool_1 = Dropout(0.2)(pool_1)

        #* layer 2
        conv_21 = Conv2D(64, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(pool_1)
        conv_22 =  Conv2D(64, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(conv_21)
        norm_2 = BatchNormalization()(conv_22)
        pool_2 = MaxPool2D()(norm_2)

        pool_2 = Dropout(0.2)(pool_2)
        
        #* layer 3
        conv_31 = Conv2D(128, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(pool_2)
        conv_32 =  Conv2D(128, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(conv_31)
        norm_3 = BatchNormalization()(conv_32)
        pool_3 = MaxPool2D()(norm_3)
        pool_3 = Dropout(0.2)(pool_3)

        flatten_img = Flatten()(pool_3)
        dence_im_1 = Dense(1024, activation = "relu")(flatten_img)
        dence_im_1 = Dropout(0.5)(dence_im_1)

        dence_img_last = Dense(output_filters, 
                                activation = "relu", 
                                name='img_last_layer')(dence_im_1)
        images_m = Model(inputs=inputs_img, outputs=dence_img_last)
        return images_m

    def _features_model(self, input_features_shape, output_filters):
        inputs_features = Input(shape=input_features_shape, name='feature_inputs')
        flatten_feat = Flatten()(inputs_features)
        dence_f_1 = Dense(512, activation = "relu", kernel_initializer='he_normal')(flatten_feat)
        dence_f_1 = Dropout(0.5)(dence_f_1)
        dence_f_2 = Dense(512, activation = "relu", kernel_initializer='he_normal')(dence_f_1)
        dence_f_2 = Dropout(0.5)(dence_f_2)

        dence_features_last = Dense(output_filters,
                                activation = "relu", 
                                name='features_last_layer')(dence_f_2)
        features_m = Model(inputs=inputs_features, outputs=dence_features_last)
        return features_m

    def _init_model_combined_(self, input_img_shape, 
                             input_features_shape, 
                             num_classes,
                             num_filters_last=32):
        #! first part
        images = self._img_model_(input_img_shape, num_filters_last)
        #! second part
        features = self._features_model(input_features_shape, num_filters_last)

        #! combine two outputs
        combined = Concatenate()([images.output, features.output])
        #dence_combined = Dense(1024, activation = "relu")(combined)
        #dence_combined = Dropout(0.5)(dence_combined)
        #model_outputs = Dense(num_classes, activation = "softmax")(dence_combined)
        model_outputs = Dense(num_classes, activation = "softmax")(combined)

        self.__model = Model(inputs=[images.input, features.input], outputs=model_outputs)

    def train_combined(self, images, features, labels, 
                    optim='adam',
                    n_epochs=50, 
                    batch_size=32, 
                    save_best_to=None):
        self.__model.compile(loss = 'categorical_crossentropy', 
                            optimizer = optim, 
                            metrics = ["accuracy"])
        if save_best_to is not None:
            history = self.__model.fit([images, features], labels, 
                                callbacks = [ModelCheckpoint(save_best_to, 
                                                    monitor = "val_acc", 
                                                    save_best_only = True, 
                                                    save_weights_only = True, 
                                                    mode = "auto")],
                                epochs = n_epochs, 
                                batch_size = batch_size,
                                verbose = 1, 
                                shuffle = True, 
                                validation_split = 0.1)
        else:
            history = self.__model.fit([images, features], labels, 
                                batch_size = batch_size,  \
                                epochs = n_epochs, 
                                verbose = 1, 
                                shuffle = True, 
                                validation_split = 0.1)
        return history
        
    def train(self, x_input, labels, 
                    optim='adam',
                    n_epochs=50, 
                    batch_size=32, 
                    save_best_to=None):
        self.__model.compile(loss = 'categorical_crossentropy', 
                            optimizer = optim, 
                            metrics = ["accuracy"])
        if save_best_to is not None:
            history = self.__model.fit(x_input, labels, 
                                callbacks = [ModelCheckpoint(save_best_to, 
                                                    monitor = "val_acc", 
                                                    save_best_only = True, 
                                                    save_weights_only = True, 
                                                    mode = "auto")],
                                epochs = n_epochs, 
                                batch_size = batch_size,
                                verbose = 1, 
                                shuffle = True, 
                                validation_split = 0.1)
        else:
            history = self.__model.fit(x_input, labels, 
                                batch_size = batch_size,  \
                                epochs = n_epochs, 
                                verbose = 1, 
                                shuffle = True, 
                                validation_split = 0.1)
        return history

    def load_model(self, json_filename):
        with open(json_filename, 'r') as json_file:
            model_json = json_file.read()
        self.__model = model_from_json(model_json)
            
    def load_weights(self, h5_filename):
        self.__model.load_weights(h5_filename)

    def save_weights(self, h5_filename):
        """Saves model weights to .h5 file"""
        self.__model.save_weights(h5_filename)

    def save_model(self, json_filename):
        """Saves model structure to json file"""
        model_json = self.__model.to_json()
        with open(json_filename, 'w') as json_file:
            json_file.write(model_json)


def plot_loss(history):    
    plt.subplot(2,2,1)
    plt.title('training loss')
    plt.plot(history.history['loss'])
    plt.subplot(2,2,2)
    plt.title('training accuracy')
    plt.plot(history.history['acc'])
    plt.subplot(2,2,3)
    plt.title('testing loss')
    plt.plot(history.history['val_loss'])
    plt.subplot(2,2,4)
    plt.title('testing accuracy')
    plt.plot(history.history['val_acc'])


def classify_emotions_combined_model(csv_filename, new_size, n_epochs=100, batch_size=32, load=False, model_id=''):
    features, labels, images = load_dataset_with_facial_features(csv_filename, include_images=True, new_size=new_size,
                    label_map={'neutral' : 0, 'anger' : 1, 'disgust' : 2, 'fear':3, 'happy':4, 'sadness':5, 'surprise':6})
    #x_data = np.reshape(x_data, (-1, 68, 2, 1))
    feature_shape = features.shape[1:]
    images_shape = (new_size[0], new_size[1], 1)
    images = np.reshape(images, (-1, new_size[0], new_size[1], 1))
    n_classes = np.unique(labels).shape[0]
    labels = np_utils.to_categorical(labels, n_classes)
    if load:
        model = Combined_Facial_Net(json_filename=os.path.join(ROOT_DIR,'saved_models\\' + model_id + 'model.json'), 
                                    h5_filename=os.path.join(ROOT_DIR,'saved_models\\' + model_id + 'model.h5'))
    else:
        model = Combined_Facial_Net(input_img_shape = images_shape, 
                                    input_features_shape = feature_shape, 
                                    num_classes = n_classes)
    history = model.train_combined(images, features, labels, 
                                    optim='adam',
                                    n_epochs=n_epochs, 
                                    batch_size=batch_size, 
                                    save_best_to=None)
    
    model.save_weights(os.path.join(ROOT_DIR,'saved_models\\' + model_id + 'model.h5'))
    model.save_model(os.path.join(ROOT_DIR,'saved_models\\' + model_id + 'model.json'))

    return history


def classify_emotions_features(csv_filename, n_epochs=100, batch_size=32, load = False):
    x_data, y_data, _ = load_dataset_with_facial_features(csv_filename,
                    label_map={'neutral' : 0, 'anger' : 1, 'disgust' : 2, 'fear':3, 'happy':4, 'sadness':5, 'surprise':6})
    #x_data = np.reshape(x_data, (-1, 68, 2, 1))
    n_classes = np.unique(y_data).shape[0]
    y_data = np_utils.to_categorical(y_data, n_classes)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)

    if load:
        with open(os.path.join(ROOT_DIR,'saved_models\\dlib_facial.json'), 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(os.path.join(ROOT_DIR,'saved_models\\dlib_facial.h5'))
    else:
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(512, activation = "relu", kernel_initializer='he_normal'))
        model.add(Dropout(0.5))     # reg
        model.add(Dense(512, activation = "relu", kernel_initializer='he_normal'))
        model.add(Dropout(0.5))     # reg
        model.add(Dense(n_classes, activation = "softmax"))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ["accuracy"])
    history_callback = model.fit(x_train, y_train, batch_size = batch_size, validation_split=0.2, \
                        epochs = n_epochs, verbose = 1, shuffle = True)

    score = model.evaluate(x_test, y_test, verbose = 0)

    print("Test score :", score[0])
    print("Test accuracy :", score[1], "\n")
    # model_json = model.to_json()
    # with open(os.path.join(ROOT_DIR,'saved_models\\dlib_facial_v2.json'), 'w') as json_file:
    #     json_file.write(model_json)
    # model.save_weights(os.path.join(ROOT_DIR,'saved_models\\dlib_facial_v2.h5'))

    return history_callback



if __name__=='__main__':
    print(os.path.abspath(os.path.join('..', 'utils')))
    history_callback = classify_emotions_features(os.path.join(ROOT_DIR,'data\\dataset.csv'), 
                                                batch_size=512, n_epochs=3000, load=False)
    #history_callback = classify_emotions_combined_model(os.path.join(ROOT_DIR,'data\\dataset.csv'), 
    #                                batch_size=64, new_size=(96, 96), 
    #                                n_epochs=300, model_id='facial_comb_')
    plot_loss(history_callback)
    plt.show()
    