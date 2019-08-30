import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, model_from_json, Model 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from load_pics import load_facial_dataset_csv, load_img, load_facial_data_kadle_cvs, load_facial_data_kadle2d
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from face_detector import faces_from_database_dnn, find_faces_dnn
from sklearn.utils import shuffle
from collections import OrderedDict
from facial_special_settings import SPECIAL_SETTINGS_KADLE_DATA
import random
import cv2

class Facial_Feature_Net:

    def __init__(self, loss_f = "mse", optim = 'rmsprop'):
        self._model = Sequential()
        self._loss_func = loss_f
        self._optim  = optim
        self.__compiled = False


    def init_model_2(self, input_shape, n_features, global_pool=False):
        
        self._model.add(BatchNormalization(input_shape = input_shape))
        
        self._model.add(Conv2D(32, (3, 3), padding = "valid", 
                                           input_shape = input_shape, 
                                           activation = 'relu', 
                                           kernel_initializer='he_normal'))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))
        self._model.add(Dropout(0.2))

        self._model.add(Conv2D(64, (3, 3), padding = "valid", 
                                           input_shape = input_shape, 
                                           activation = 'relu'))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), 
                                           padding = "valid"))
        self._model.add(Dropout(0.2))

        self._model.add(Conv2D(128, (3, 3), padding = "valid", 
                                           input_shape = input_shape, 
                                           activation = 'relu'))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), 
                                           padding = "valid"))
        self._model.add(Dropout(0.2))

        self._model.add(Conv2D(256, (3, 3), padding = "valid", 
                                            input_shape = input_shape, 
                                            activation = 'relu'))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), 
                                            padding = "valid"))
        self._model.add(Dropout(0.2))

        self._model.add(Conv2D(512, (3, 3), padding = "valid", 
                                            input_shape = input_shape, 
                                            activation = 'relu'))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), 
                                            padding = "valid"))
        self._model.add(Dropout(0.2))

        if global_pool:
            self._model.add(GlobalAveragePooling2D())
        else:
            self._model.add(Flatten())
        
        self._model.add(Dense(512, activation = "relu", 
                                   kernel_initializer='he_normal'))
        self._model.add(Dense(512, activation = "relu", 
                                   kernel_initializer='he_normal'))
        self._model.add(Dropout(0.5))     # reg
        
        self._model.add(Dense(1024, activation = "relu", 
                                   kernel_initializer='he_normal'))
        self._model.add(Dense(1024, activation = "relu", 
                                   kernel_initializer='he_normal'))
        self._model.add(Dropout(0.5))     # reg

        self._model.add(Dense(n_features))


    def init_model(self, input_shape, n_features):
        self._model.add(BatchNormalization(input_shape = input_shape))

        self._model.add(Conv2D(24, (5, 5), padding = "same", input_shape = input_shape, 
                            activation = 'sigmoid', kernel_initializer='he_normal'))
        self._model.add(BatchNormalization(input_shape = input_shape))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))
        
        self._model.add(Conv2D(36, (5, 5), padding = "same", input_shape = input_shape, activation = 'sigmoid'))
        self._model.add(Conv2D(36, (5, 5), padding = "same", input_shape = input_shape, activation = 'sigmoid'))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))
        
        #self._model.add(Conv2D(48, (5, 5), padding = "same", input_shape = input_shape, activation = 'relu'))
        #self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))
        
        self._model.add(Conv2D(48, (3, 3), padding = "same", input_shape = input_shape, activation = 'sigmoid'))
        self._model.add(BatchNormalization(input_shape = input_shape))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))
        
        self._model.add(Conv2D(64, (3, 3), padding = "same", input_shape = input_shape, activation = 'sigmoid'))
        self._model.add(GlobalAveragePooling2D())
        
        self._model.add(Dense(512, activation = "sigmoid", kernel_initializer='he_normal'))
        self._model.add(Dropout(0.5))     # reg
        self._model.add(Dense(256, activation = "sigmoid", kernel_initializer='he_normal'))
        self._model.add(Dropout(0.5))     # reg
        self._model.add(Dense(n_features))


    def train(self, x_train, y_train, 
                    batch_size=16, n_epochs=50, loss_func="mse", 
                    optim='rmsprop', save_best=True, save_best_to="model.hdf5"):
        """Compile and train the model"""

        # optimizing
        self._loss_func = loss_func
        self._optim  = optim
        self._model.compile(loss = loss_func, optimizer = optim, metrics = ["accuracy"])
        self.__compiled = True

        # learning
        if save_best:
            history_callback = self._model.fit(x_train, y_train, batch_size = batch_size, validation_split=0.2,  \
                        callbacks = [ModelCheckpoint(save_best_to, monitor = "val_acc", save_best_only = True, save_weights_only = True, mode = "auto")], \
                        epochs = n_epochs, verbose = 1, shuffle = True)
        else:
            history_callback = self._model.fit(x_train, y_train, batch_size = batch_size, validation_split=0.2, \
                        epochs = n_epochs, verbose = 1, shuffle = True)
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


def find_features(img, model, new_size, conf_threshold = 0.97):
    detections = find_faces_dnn(img)
    (h, w) = img.shape[:2]
    faces = []
    bboxes = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence >= conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = img[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            if new_size is not None:
                assert len(new_size) == 2
                face = cv2.resize(face, new_size, interpolation = cv2.INTER_AREA)
            faces.append(np.resize(face, (face.shape[0], face.shape[1], 1)))
            bboxes.append([(startX, startY), (endX, endY)])

    faces = np.array(faces, dtype='float32')
    faces = faces / 123.0 - 1
    face_features = model.predict(faces)
    
    bboxes = np.array(bboxes).reshape(faces.shape[0], 2, 2)
    for i in range(faces.shape[0]):
        for j in range(0, face_features[i].shape[0] - 1, 2):
            org_h = bboxes[i, 1, 1] - bboxes[i, 0, 1] 
            org_w = bboxes[i, 1, 0] - bboxes[i, 0, 0] 
            img = cv2.circle(img, (int(face_features[i][j] * org_h + bboxes[i, 1, 0]), 
                                    int(face_features[i][j + 1] * org_w + bboxes[i, 1, 1])), 
                                     2, (0, 0, 255), 1)

    return img


def detect_and_find_features(img, model, conf_threshold = 0.97, new_size=(144, 144)):
    detections = find_faces_dnn(img)
    (h, w) = img.shape[:2]
    faces = []
    bboxes = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence >= conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            if (box >= [0., 0., 0., 0.]).all()  and (box <= [w, h, w, h]).all():
                (startX, startY, endX, endY) = box.astype("int")
                face = img[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                if new_size is not None:
                    assert len(new_size) == 2
                    face = cv2.resize(face, new_size, interpolation = cv2.INTER_AREA)
                faces.append(np.resize(face, (face.shape[0], face.shape[1], 1)))
                bboxes.append([(startX, startY), (endX, endY)])

    faces = np.array(faces, dtype='float32')
    if faces.shape[0] == 0:
        return None
    faces = faces / 123.0 - 1
    face_features = model.predict(faces)

    bboxes = np.array(bboxes).reshape(faces.shape[0], 2, 2)
    faces_img = []
    for i in range(faces.shape[0]):
        faces_img.append(cv2.cvtColor(faces[i], cv2.COLOR_GRAY2BGR))
        for j in range(0, face_features[i].shape[0] - 1, 2):
            org_h = bboxes[i, 1, 1] - bboxes[i, 0, 1] 
            org_w = bboxes[i, 1, 0] - bboxes[i, 0, 0] 
            faces_img[i] = cv2.resize(faces_img[i], (new_size[0]*3, new_size[1]*3), interpolation = cv2.INTER_AREA)
            faces_img[i] = cv2.circle(faces_img[i], (int(face_features[i][j] * 5 * new_size[0]), 
                                    int(face_features[i][j + 1] * 5 * new_size[1])), 
                                     2, (0, 0, 255), 1)

    return faces_img


def plot_loss(hist, name, plt, rmse=False):
    '''
    rmse: if True, then rmse is plotted with original scale 
    '''
    loss = hist['loss']
    val_loss = hist['val_loss']
    if rmse:
        loss = np.sqrt(np.array(loss))*48 
        val_loss = np.sqrt(np.array(val_loss))*48 
        
    plt.plot(loss, "--", linewidth=3, label="train:" + name)
    plt.plot(val_loss, linewidth=3, label="val:" + name)


def plot_hist(history_call):
    plt.subplot(2,2,1)
    plt.title('training loss')
    plt.plot(history_call.history['loss'])
    plt.subplot(2,2,2)
    plt.title('training accuracy')
    plt.plot(history_call.history['acc'])
    plt.subplot(2,2,3)
    plt.title('testing loss')
    plt.plot(history_call.history['val_loss'])
    plt.subplot(2,2,4)
    plt.title('testing accuracy')
    plt.plot(history_call.history['val_acc'])
    plt.show()


def train_features(load_model=False, num_features=76, model_id=''):
    im_rows, im_cols, channels = 96, 96, 1
    if num_features == 76:
        x_data, y_data = load_facial_dataset_csv('data\\muct\\imgs\\', 'data\\muct\\muct76_bbox.csv', True, (im_rows, im_cols))
    elif num_features == 15:
        x_data, y_data = load_facial_data_kadle2d('data\\facial_features\\training.csv')
    else:
        return None
    x_data = np.array(x_data, dtype='float32')
    y_data = np.array(y_data, dtype='float32')

    n_features = y_data.shape[1]
    x_data = x_data/255.0   
    x_data = x_data.reshape(x_data.shape[0], im_rows, im_cols, channels)

    Model = Facial_Feature_Net()
    if load_model:
        Model.load_model("models\\facial_model.json")
        Model.load_weights("models\\facial_model.h5")
    else:
        Model.init_model_2(input_shape=(im_rows, im_cols, channels), n_features=n_features)

    history_call = Model.train(x_data, y_data, 
                                batch_size=32, 
                                n_epochs=100, 
                                optim = 'rmsprop',
                                save_best = False,
                                save_best_to="models\\facial_model.h5")

    #Model.evaluate_accur(x_test, y_test)
    Model.save_model("models\\" + model_id + "facial_model.json")
    Model.save_weights("models\\" + model_id + "facial_model.h5")
    
    plot_hist(history_call)
    return Model


def train_features_special(load_model=False):
    special_ft = OrderedDict()
    im_rows, im_cols, channels = 96, 96, 1

    for setting in SPECIAL_SETTINGS_KADLE_DATA:
        cols = setting['columns']
        flip_indices = setting['flip_indices']

        x_data, y_data = load_facial_data_kadle2d('data\\facial_features\\training.csv', cols)
        x_data = np.array(x_data, dtype='float32')
        n_features = y_data.shape[1]
        x_data /= 255.   
        Model = Facial_Feature_Net()
        if load_model:
            Model.load_model("models\\" + setting['save_as'] + "_facial_model.json")
            Model.load_weights("models\\" + setting['save_as'] + "_facial_model.h5")
        else:
            Model.init_model_2(input_shape=(im_rows, im_cols, channels), n_features=n_features)

        print("Training features: ", cols)
        history_call = Model.train(x_data, y_data, 
                                    batch_size=32, 
                                    n_epochs=70, 
                                    optim = 'adam',
                                    save_best = False,
                                    save_best_to="models\\facial_model.h5")

        special_ft[cols] = {"model":Model,
                             "hist":history_call}

        Model.save_model("models\\special_features\\" + setting['save_as'] + "_facial_model.json")
        Model.save_weights("models\\special_features\\" + setting['save_as'] + "_facial_model.h5")
    
    #plot_hist(history_call)
    return (special_ft)



if __name__=="__main__":
    random.seed()
    train_features_special()

    # images = ['data\\kanade\\cohn-kanade-images\\S022\\004\\S022_004_00000006.png',
    #         'data\\muct\\imgs\\i000qb-fn.jpg', 'data\\muct\\imgs\\i003sc-fn.jpg',
    #         'data\\muct\\imgs\\i012rb-mn.jpg', 'data\\muct\\imgs\\i031sd-fn.jpg']
    # for img_name in images:
    #     img = load_img(img_name, False, None)
    #     faces_found = find_features(img, Model, new_size=(im_rows, im_cols))
    #     cv2.imshow(img_name, faces_found)
    #     cv2.waitKey(0)
    #     #for face in faces_found:
    #     #    cv2.imshow(img_name, face)
    #     #    cv2.waitKey(0) 