import tensorflow as tf
import numpy as np
from keras.metrics import MAE
from keras.optimizers import SGD, Adam, RMSprop
from keras.activations import relu
from keras.models import Sequential, model_from_json, Model 
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from load_pics import load_img, load_facial_data_kadle_cvs, load_facial_data_kadle2d, load_facial_dataset_for_autoencoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from face_detector import faces_from_database_dnn, find_faces_dnn
from sklearn.utils import shuffle
from collections import OrderedDict
from facial_special_settings import SPECIAL_SETTINGS_KADLE_DATA
import random
import cv2
import pandas as pd

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

    def init_model_3(self, input_shape, n_features):
        self._model.add(Conv2D(32, (3, 3), padding = "same", 
                                           input_shape = input_shape, 
                                           activation = 'relu', 
                                           kernel_initializer='he_normal'))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))

        self._model.add(Conv2D(64, (3, 3), padding = "same", 
                                           input_shape = input_shape, 
                                           activation = 'relu'))
        self._model.add(Conv2D(64, (3, 3), padding = "same", 
                                           input_shape = input_shape, 
                                           activation = 'relu'))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))

        self._model.add(Conv2D(64, (3, 3), padding = "same", 
                                           input_shape = input_shape, 
                                           activation = 'relu'))
        self._model.add(Conv2D(64, (3, 3), padding = "same", 
                                           input_shape = input_shape, 
                                           activation = 'relu'))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))
        self._model.add(BatchNormalization())
        self._model.add(Dropout(0.2))

        self._model.add(Conv2D(128, (3, 3), padding = "same", 
                                           input_shape = input_shape, 
                                           activation = 'relu'))
        self._model.add(Conv2D(128, (3, 3), padding = "same", 
                                           input_shape = input_shape, 
                                           activation = 'relu'))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))

        self._model.add(Conv2D(256, (3, 3), padding = "same", 
                                           input_shape = input_shape, 
                                           activation = 'relu'))
        self._model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid"))
        self._model.add(BatchNormalization())
        self._model.add(Dropout(0.2))            
        
        self._model.add(Flatten())
        
        self._model.add(Dropout(0.5))     # reg
        self._model.add(Dense(1024, activation = "relu", 
                                   kernel_initializer='he_normal'))
        self._model.add(Dropout(0.5))     # reg        
        self._model.add(Dense(1024, activation = "relu", 
                                   kernel_initializer='he_normal'))
        self._model.add(Dropout(0.5))     # reg
        self._model.add(Dense(n_features))

    def init_resnet50(self, input_shape, n_features):
        model = ResNet50(weights = "imagenet", include_top=False, input_shape = input_shape)
        x = model.output
        
        x = Flatten()(x)
        x = Dense(2048, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)

        predictions = Dense(n_features)(x)
        self._model =  Model(input = model.input, output = predictions)

    def train(self, x_train, y_train, 
                    batch_size=16, n_epochs=50, loss_func="mse", 
                    optim='rmsprop', save_best=True, save_best_to="model.hdf5"):
        """Compile and train the model"""

        # optimizing
        self._loss_func = loss_func
        self._optim  = optim
        self._model.compile(loss = loss_func, optimizer = optim, metrics = ['mae'])
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
            self._model.compile(loss = self._loss_func, optimizer = self._optim, metrics = [MAE])
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


class Autoencoder_Facial_Net:
    def __init__(self):
        self._model = None
        self.__compiled = False

    def init_model(self, input_shape,
                            base_filters_num=32, 
                            use_batch_norm=True, 
                            use_concat=True):

        def conv_block(layer, filters, k_size, strides, use_b_norm):
            op = Conv2D(filters, (k_size, k_size), padding = "same", 
                                           strides=(strides, strides),
                                           input_shape = input_shape, 
                                           activation='relu',
                                           kernel_initializer='he_normal')(layer)
            if use_b_norm:
                op = BatchNormalization()(op)
            return op

        inputs = Input(shape=input_shape, name='inputs')

        c1=conv_block(inputs, base_filters_num, k_size=3, strides=2, use_b_norm=use_batch_norm)
        c2=conv_block(c1, base_filters_num*2, k_size=3, strides=2, use_b_norm=use_batch_norm)
        c3=conv_block(c2, base_filters_num*4, k_size=3, strides=2, use_b_norm=use_batch_norm)
        c4=conv_block(c3, base_filters_num*8, k_size=3, strides=2, use_b_norm=use_batch_norm)

        u1 = UpSampling2D()(c4)
        if use_concat:
                u1 = Concatenate()([u1, c3])
        u1 = conv_block(u1, base_filters_num*8, k_size=3, strides=1, use_b_norm=use_batch_norm)

        u2 = UpSampling2D()(u1)
        if use_concat:
                u2 = Concatenate()([u2, c2])
        u2 = conv_block(u2, base_filters_num*4, k_size=3, strides=1, use_b_norm=use_batch_norm)

        u3 = UpSampling2D()(u2)
        if use_concat:
                u3 = Concatenate()([u3, c1])
        u3 = conv_block(u3, base_filters_num*2, k_size=3, strides=1, use_b_norm=use_batch_norm)

        u4 = UpSampling2D()(u3)
        u4 = conv_block(u4, base_filters_num, k_size=3, strides=1, use_b_norm=use_batch_norm)
        
        output = Conv2D(1, (1, 1), strides=(1, 1), activation='sigmoid', name='logits')(u4)
        #print(inputs.shape)
        #print(output.shape)
        self._model = Model(input=inputs, output=output)

    def __compile(self, optim):
        def loss_weighted(y_true, y_predicted):
            return tf.reduce_mean(
                                #tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                #                                        logits=y_predicted)
                                tf.nn.weighted_cross_entropy_with_logits(targets=y_true,
                                                                         logits=y_predicted,
                                                                         pos_weight=0.1)
                                )
        self._model.compile(optimizer=optim, loss='mse')
        self.__compiled = True

    def train(self, x_data, y_data, optim='adam', batch_size=32, n_epochs=50):
        self.__compile(optim)

        history_callback = self._model.fit(x_data, y_data, 
                                    batch_size = batch_size,  
                                    epochs = n_epochs, 
                                    verbose = 1, shuffle = True, 
                                    validation_split=0.2)
        return history_callback


    def save_model(self, json_filename, h5_filename):
        model_json = self._model.to_json()
        with open(json_filename, 'w') as json_file:
             json_file.write(model_json)
        self._model.save_weights(h5_filename)


    def load_model(self, json_filename, h5_filename):
        with open(json_filename, 'r') as json_file:
            model_json = json_file.read()
        self._model = model_from_json(model_json)
        self._model.load_weights(h5_filename)
        self.__compile('adam')


    def predict(self, x):
        if self.__compiled:
            return self._model.predict(x)
        else:
            return None


def warp(x, y, matr):
        x_ = x*matr[0][0] + y*matr[0][1] + matr[0][2]
        y_ = x*matr[1][0] + y*matr[1][1] + matr[1][2]
        return x_, y_

def warp_point(p, matr):
    x, y = p
    x_ = x*matr[0][0] + y*matr[0][1] + matr[0][2]
    y_ = x*matr[1][0] + y*matr[1][1] + matr[1][2]
    return np.array([x_, y_])


def apply_transform(x_set, y_set, matr, feature_size):
        for i in range(x_set.shape[0]):
                x_set[i] = cv2.warpAffine(x_set[i], matr, (x_set.shape[1], x_set.shape[2]))
                for j in range(feature_size):
                    y_set[i][j] = warp(y_set[i][j][0], y_set[i][j][1], matr)
        return x_set, y_set


def augment_dataset(x_data, y_data, feature_size=76):
    x_flip, x_rot, y_flip, y_rot = train_test_split(np.array(x_data), 
                                        np.reshape(y_data, (-1, feature_size, 2)), 
                                        test_size=0.5, random_state=random.seed())
    x_rot_right, x_rot_left, y_rot_right, y_rot_left = train_test_split(x_rot, y_rot, 
                                        test_size=0.5, random_state=random.seed())
    matr_flip = np.array([[-1, 0, 0], [0, 1, 0]], dtype='float32')
    matr_rot_right = np.array([[0.4, 0, 0], [0, 1, 0]], dtype='float32')
    matr_rot_left = np.array([[0.75, 0, 0], [0, 1, 0]], dtype='float32')
    x_flip, y_flip = apply_transform(x_flip, y_flip, matr_flip, feature_size)
    x_rot_right, y_rot_right = apply_transform(x_rot_right, y_rot_right, matr_rot_right, feature_size)
    x_rot_left, y_rot_left = apply_transform(x_rot_left, y_rot_left, matr_rot_left, feature_size)
    res_x = np.concatenate((x_rot_right, x_rot_left, x_flip))
    res_y = np.concatenate((y_rot_right, y_rot_left, y_flip))
    return res_x, res_y

def get_face(img, greyscale, new_size, row):
    if greyscale:
            im = img[row['bbox_y0']:row['bbox_y1'], row['bbox_x0']:row['bbox_x1']]
    else:
            im = img[row['bbox_y0']:row['bbox_y1'], row['bbox_x0']:row['bbox_x1'], :]
    if new_size is not None:
            im = cv2.resize(im, new_size, interpolation = cv2.INTER_AREA)
    return im


def load_facial_dataset(img_dir, csv_filename, greyscale=True, new_size=None, augment=False):
    # bbox_x0,bbox_y0,bbox_x1,bbox_y1
    df = pd.read_csv(csv_filename)
    x_data, y_data = [], []
    
    for index, row in df.iterrows():
        img = load_img(img_dir + row['name'] + '.jpg', greyscale, None)
        im = get_face(img, greyscale, new_size, row)
        coords = np.array(row.iloc[3:-4].values).reshape(76, 2)
        w, h = row['bbox_x1'] -  row['bbox_x0'], row['bbox_y1'] -  row['bbox_y0']
        w_org = 2 / w
        h_org = 2 / h
        center = (w / 2, h / 2) 
        coords = (coords - [row['bbox_x0'], row['bbox_y0']]) 
        if augment:
            matr_flip = np.array([[-1, 0, w], 
                                [0, 1, 0]], 
                                dtype='float32')
            matr_rot_right = cv2.getRotationMatrix2D(center, 30, 1.0) 
            matr_rot_left = cv2.getRotationMatrix2D(center, -30, 1.0) 

            fliped_im = cv2.flip(im, 0 )
            coords_fliped = np.array([warp_point(point, matr_flip) for point in coords])            
            
            rot_right_im = cv2.warpAffine(im, matr_rot_right, (im.shape[0], im.shape[1]))
            coords_rot_right = np.array([warp_point(point, matr_rot_right) for point in coords])            
            
            rot_left_im = cv2.warpAffine(im, matr_rot_left, (im.shape[0], im.shape[1]))
            coords_rot_left = np.array([warp_point(point, matr_rot_left) for point in coords])

            coords_fliped = coords_fliped * [w_org, h_org] - 1
            coords_rot_right = coords_rot_right * [w_org, h_org] - 1
            coords_rot_left = coords_rot_left * [w_org, h_org] - 1

            x_data.append(fliped_im)
            y_data.append(coords_fliped)        
            x_data.append(rot_right_im)
            y_data.append(coords_rot_right)        
            x_data.append(rot_left_im)
            y_data.append(coords_rot_left)
        coords = coords * [w_org, h_org] - 1
        x_data.append(im)
        y_data.append(coords)
    x_data, y_data = shuffle(x_data, y_data, random_state=42)
    return x_data, y_data


def flip_dataset(x_data, y_data, feature_size=76, pers=0.5):
    x_flip, x_rot, y_flip, y_rot = train_test_split(np.array(x_data), 
                                        np.reshape(y_data, (-1, feature_size, 2)), 
                                        test_size=pers, random_state=random.seed())
    matr_flip = np.array([[-1, 0, 0], [0, 1, 0]], dtype='float32')
    x_flip, y_flip = apply_transform(x_flip, y_flip, matr_flip, feature_size)
    return x_flip, np.reshape(y_flip, (-1, feature_size*2))


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


def predict_mask(img, model, new_size, conf_threshold = 0.97):
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
    faces = faces / 255.0
    face_features = model.predict(faces)

    return face_features


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
    faces = faces / 255.0
    face_features = model.predict(faces)

    bboxes = np.array(bboxes).reshape(faces.shape[0], 2, 2)
    faces_img = []
    for i in range(faces.shape[0]):
        faces_img.append(cv2.cvtColor(faces[i], cv2.COLOR_GRAY2BGR))
        for j in range(0, face_features[i].shape[0] - 1, 2):
            org_h = bboxes[i, 1, 1] - bboxes[i, 0, 1] 
            org_w = bboxes[i, 1, 0] - bboxes[i, 0, 0] 
            new_h, new_w = new_size[0]*5, new_size[1]*5
            faces_img[i] = cv2.resize(faces_img[i], (new_h, new_w), interpolation = cv2.INTER_AREA)
            faces_img[i] = cv2.circle(faces_img[i], (int((face_features[i][j] + 1) * new_w /2), 
                                    int((face_features[i][j + 1] + 1) * new_h / 2)), 
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
    plt.subplot(2,1,1)
    plt.title('training loss')
    plt.plot(history_call.history['loss'])
    plt.subplot(2,1,2)
    plt.title('testing loss')
    plt.plot(history_call.history['val_loss'])
    plt.show()


def train_features(load_model=False, num_features=76, model_id=''):
    im_rows, im_cols, channels = 96, 96, 1
    if num_features == 76:
        x_data, y_data = load_facial_dataset('data\\muct\\imgs\\', 
                                            'data\\muct\\muct76_bbox.csv', 
                                            True, (im_rows, im_cols), augment=False)
    elif num_features == 15:
        x_data, y_data = load_facial_data_kadle2d('data\\facial_features\\training.csv')
    else:
        return None
    x_data = np.array(x_data, dtype='float32')
    y_data = np.array(y_data, dtype='float32')
    #x_augmented, y_augmented = flip_dataset(x_data, y_data, num_features)
    #x_data = np.concatenate((x_data, x_augmented))
    #y_data = np.concatenate((y_data, y_augmented))
    x_data = np.reshape(x_data, (-1, im_rows, im_cols, channels))
    y_data = np.reshape(y_data,(-1, num_features*2))
    n_features = y_data.shape[1]
    x_data = x_data/255.0   
    Model = Facial_Feature_Net()

    if load_model:
        Model.load_model("models\\" + model_id + "facial_model.json")
        Model.load_weights("models\\" + model_id + "facial_model.h5")
    else:
        Model.init_model_3(input_shape=(im_rows, im_cols, channels), n_features=n_features)

    history_call = Model.train(x_data, y_data, 
                                batch_size=64, 
                                n_epochs=50, 
                                optim = 'rmsprop',
                                save_best = False,
                                save_best_to="models\\" + model_id + "facial_model.h5")

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


def train_features_auto(load_model=False, num_features=76, model_id=''):
    im_rows, im_cols, channels = 96, 96, 1
    num_features = 76
    x_data, y_data = load_facial_dataset_for_autoencoder('data\\muct\\imgs\\', 'data\\muct\\muct76_bbox.csv', True, (im_rows, im_cols))
    x_data = np.array(x_data, dtype='float32')
    y_data = np.array(y_data, dtype='float32')
    #x_augmented, y_augmented = flip_dataset(x_data, y_data, num_features)
    #x_data = np.concatenate((x_data, x_augmented))
    #y_data = np.concatenate((y_data, y_augmented))
    x_data = x_data.reshape(-1, im_rows, im_cols, channels)
    y_data = x_data.reshape(-1, im_rows, im_cols, 1)
    x_data = x_data/255.0   
    Model = Autoencoder_Facial_Net()
    if load_model:
        Model.load_model("models\\" + model_id + "facial_model.json", "models\\" + model_id + "facial_model.h5")
    else:
        Model.init_model(input_shape=(im_rows, im_cols, channels))

    history_call = Model.train(x_data, y_data, 
                                batch_size=64, 
                                n_epochs=50, 
                                optim = 'adam')

    #Model.evaluate_accur(x_test, y_test)
    Model.save_model("models\\" + model_id + "auto_facial_model.json", "models\\" + model_id + "auto_facial_model.h5")
    
    #plot_hist(history_call)
    return Model

if __name__=="__main__":
    random.seed()
    train_features(load_model=False, model_id='76_')
    #train_features_special()

    images = ['data\\kanade\\cohn-kanade-images\\S022\\004\\S022_004_00000006.png',
            'data\\muct\\imgs\\i000qb-fn.jpg', 'data\\muct\\imgs\\i003sc-fn.jpg',
            'data\\muct\\imgs\\i012rb-mn.jpg', 'data\\muct\\imgs\\i031sd-fn.jpg']
    # im_rows, im_cols, channels = 96, 96, 1
    # for img_name in images:
    #     img = load_img(img_name, False, None)
    #     faces_found = predict_mask(img, Model, new_size=(im_rows, im_cols))
    #     for face in faces_found:
    #         cv2.imshow(img_name, face)
    #         cv2.waitKey(0)
        #for face in faces_found:
        #    cv2.imshow(img_name, face)
        #    cv2.waitKey(0) 