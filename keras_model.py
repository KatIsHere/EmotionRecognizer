import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, Adam
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

class Emotion_Net:

    def __init__(self, input_shape, n_classes):
        self._model = Sequential()
        self.__arcitecture_2(input_shape, n_classes)

    def __arcitecture_1(self, input_shape, n_classes):
        self._model.add(Conv2D(32, (7, 7), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(Conv2D(64, (5, 5), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization(epsilon=0.0001))
        self._model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"))

        #self._model.add(Conv2D(64, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        #self._model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"))

        self._model.add(Conv2D(128, (5, 5), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization(epsilon=0.0001))
        self._model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"))

        self._model.add(Dropout(0.4))

        self._model.add(Conv2D(256, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization(epsilon=0.0001))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        self._model.add(Conv2D(256, (1, 1), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization(epsilon=0.0001))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        #self._model.add(Dropout(0.3)

        self._model.add(Conv2D(512, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization(epsilon=0.0001))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        # tensor reforming 
        self._model.add(Flatten())
        self._model.add(Dense(1024, activation = "relu"))
        self._model.add(Dropout(0.3))     # reg
        self._model.add(Dense(2048, activation = "relu"))
        self._model.add(Dropout(0.5))     # reg
        self._model.add(Dense(n_classes, activation = 'softmax'))


    def __arcitecture_2(self, input_shape, n_classes):

        self._model.add(Conv2D(32, (7, 7), padding = "valid", input_shape = input_shape, activation = 'relu'))
        #self._model.add(Conv2D(64, (7, 7), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization())       
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))        
        
        self._model.add(Conv2D(64, (5, 5), padding = "valid", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization())
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        #self._model.add(Conv2D(64, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        #self._model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"))

        self._model.add(Conv2D(256, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization())
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        self._model.add(Dropout(0.3))

        self._model.add(Conv2D(128, (1, 1), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization())
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        self._model.add(Conv2D(64, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization())
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        #self._model.add(Dropout(0.3)

        #self._model.add(Conv2D(512, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        #self._model.add(BatchNormalization(epsilon=0.0001))
        #self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        # tensor reforming 
        self._model.add(Flatten())
        self._model.add(Dense(1024, activation = "relu"))
        self._model.add(Dropout(0.4))     # reg
        self._model.add(Dense(512, activation = "relu"))
        self._model.add(Dropout(0.5))     # reg
        self._model.add(Dense(n_classes, activation = 'softmax'))


    def train(self, x_train, y_train, x_test, y_test, 
                    batch_size=16, n_epochs=50, loss_func="categorical_crossentropy", 
                    optim=SGD(lr=0.001), save_best=True, save_best_to="model.hdf5"):
        """Compile and train the model"""

        # optimizing
        self._model.compile(loss = loss_func, optimizer = optim, metrics = ["accuracy"])

        # learning
        if save_best:
            self._model.fit(x_train, y_train, batch_size = batch_size,  \
                        callbacks = [ModelCheckpoint(save_best_to, monitor = "val_acc", save_best_only = True, save_weights_only = True, mode = "auto")], \
                        epochs = n_epochs, verbose = 1, shuffle = True, validation_data = (x_test, y_test))
        else:
            self._model.fit(x_train, y_train, batch_size = batch_size,  \
                        epochs = n_epochs, verbose = 1, shuffle = True, validation_data = (x_test, y_test))


    def evaluate_accur(self, x_test, y_test):
        # print accuracy
        score = self._model.evaluate(x_test, y_test, verbose = 0)
        print("Test score :", score[0])
        print("Test accuracy :", score[1], "\n")


    def save_weights(self, h5_filename):
        """Saves model weights to .h5 file"""
        self._model.save_weights(h5_filename)


    def save_model(self, json_filename):
        """Saves model structure to json file"""
        model_json = self._model.to_json()
        with open(json_filename, 'w') as json_file:
            json_file.write(model_json)


    def load_weights(self, h5_filename):
        """Loads weights from .h5 file"""
        self._model.load_weights(h5_filename)


    def load_model(self, json_filename, loss_func="categorical_crossentropy", optim=SGD(lr=0.001)):
        """Loads model structure from json file"""
        with open(json_filename, 'r') as json_file:
            model_json = json_file.read()
        self._model = model_from_json(model_json)
        self._model.compile(loss = loss_func, optimizer = optim, metrics = ["accuracy"])
    