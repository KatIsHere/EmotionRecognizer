import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, Adam
from keras.models import Sequential, model_from_json, Model 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
from scipy.io import loadmat

class Emotion_Net:

    def __init__(self, loss_f = "categorical_crossentropy", optim = SGD(lr=0.001)):
        self._model = Sequential()
        self._loss_func = loss_f
        self._optim  = optim
        self.__compiled = False

    def __transfer_vgg16(self, input_shape, nb_classes):
        model = VGG16(weights = "imagenet", include_top=False, input_shape = input_shape)
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(nb_classes, activation="softmax")(x)
        self._model =  Model(input = model.input, output = predictions)

    def __transfer_dence121(self, input_shape, nb_classes):
        model = DenseNet121(weights = "imagenet", include_top=False, input_shape = input_shape)
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(nb_classes, activation="softmax")(x)
        self._model =  Model(input = model.input, output = predictions)

    def __transfer_resnet50(self, input_shape, nb_classes):
        model = ResNet50(weights = "imagenet", include_top=False, input_shape = input_shape)
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(nb_classes, activation="softmax")(x)
        self._model =  Model(input = model.input, output = predictions)

    def __transfer_vgg_face(self, input_shape, nb_classes):
        data = loadmat('vgg_face_matconvnet/data/vgg_face.mat',
               matlab_compatible=False,
               struct_as_record=False)
        net = data['net'][0,0]
        l = net.layers
        description = net.classes[0,0].description  



    def init_model(self, input_shape, n_classes, arc=0):
        if arc==0:
            self.__arcitecture_2(input_shape, n_classes)
        elif arc==1:
            self.__transfer_vgg16(input_shape, n_classes)
        elif arc==2:
            self.__transfer_resnet50(input_shape, n_classes)

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
        self._model.add(Dense(n_classes, activation = 'softmax'))


    def __arcitecture_2(self, input_shape, n_classes):

        self._model.add(Conv2D(64, (5, 5), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization())       
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))        
        
        self._model.add(Conv2D(128, (5, 5), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization())
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        #self._model.add(Conv2D(64, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        #self._model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"))

        self._model.add(Conv2D(256, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization())
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        self._model.add(Dropout(0.5))

        #self._model.add(Conv2D(128, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        #self._model.add(BatchNormalization())
        #self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        self._model.add(Conv2D(128, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        self._model.add(BatchNormalization())
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        #self._model.add(Dropout(0.3)

        #self._model.add(Conv2D(512, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
        #self._model.add(BatchNormalization(epsilon=0.0001))
        #self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same"))

        # tensor reforming 
        self._model.add(Flatten())
        self._model.add(Dense(1024, activation = "relu"))
        self._model.add(Dropout(0.5))     # reg
        self._model.add(Dense(512, activation = "relu"))
        self._model.add(Dense(n_classes, activation = 'softmax'))


    def train(self, x_train, y_train, x_test, y_test, 
                    batch_size=16, n_epochs=50, loss_func="categorical_crossentropy", 
                    optim=SGD(lr=0.001), save_best=True, save_best_to="model.hdf5"):
        """Compile and train the model"""

        # optimizing
        self._loss_func = loss_func
        self._optim  = optim
        self._model.compile(loss = loss_func, optimizer = optim, metrics = ["accuracy"])
        self.__compiled = True

        # learning
        if save_best:
            history_callback = self._model.fit(x_train, y_train, batch_size = batch_size,  \
                        callbacks = [ModelCheckpoint(save_best_to, monitor = "val_acc", save_best_only = True, save_weights_only = True, mode = "auto")], \
                        epochs = n_epochs, verbose = 1, shuffle = True, validation_data = (x_test, y_test))
        else:
            history_callback = self._model.fit(x_train, y_train, batch_size = batch_size,  \
                        epochs = n_epochs, verbose = 1, shuffle = True, validation_data = (x_test, y_test))
        return history_callback


    def augment_and_train(self, x_train, y_train, x_test, y_test, 
                    batch_size=16, n_epochs=50, loss_func="categorical_crossentropy", 
                    optim=SGD(lr=0.001), save_best=True, save_best_to="model.hdf5"):
        """Compile and train the model"""
        # generator for augmenting data
        datagen = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

        datagen.fit(x_train)

        # optimizing
        self._loss_func = loss_func
        self._optim  = optim
        self._model.compile(loss = loss_func, optimizer = optim, metrics = ["accuracy"])
        self.__compiled = True

        # learning
        if save_best:
            history_callback = self._model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), 
                        steps_per_epoch=len(x_train) // batch_size,  
                        callbacks = [ModelCheckpoint(save_best_to, monitor = "val_acc",         
                                                                   save_best_only = True,       
                                                                   save_weights_only = True,    
                                                                   mode = "auto")], 
                        epochs = n_epochs, 
                        verbose = 1, 
                        shuffle = True, 
                        validation_data = (x_test, y_test))
        else:
            history_callback = self._model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), 
                        steps_per_epoch=len(x_train) // batch_size,  
                        epochs = n_epochs, 
                        verbose = 1, 
                        shuffle = True, 
                        validation_data = (x_test, y_test))
        return history_callback


    def evaluate_accur(self, x_test, y_test):
        # print accuracy
        if not self.__compiled:
            self.__compiled = True
            self._model.compile(loss = self._loss_func, optimizer = self._optim, metrics = ["accuracy"])
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


    def load_model(self, json_filename, compile=True):
        """Loads model structure from json file"""
        with open(json_filename, 'r') as json_file:
            model_json = json_file.read()
        self._model = model_from_json(model_json)
        if compile:
            self.__compiled = True
            self._model.compile(loss = self._loss_func, optimizer = self._optim, metrics = ["accuracy"])

    def predict(self, imgs):
        if not self.__compiled:
            self.__compiled = True
            self._model.compile(loss = self._loss_func, optimizer = self._optim, metrics = ["accuracy"])
        return self._model.predict(imgs)
    