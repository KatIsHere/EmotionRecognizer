import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, model_from_json, Model 
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from facial_feature_extractor import plot_hist
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random
import cv2
import dlib
from load_pics import load_img
import pandas as pd
from facial_feature_extractor import Facial_Feature_Net

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
            self.__init_model__(input_img_shape, 
                             input_features_shape, 
                             num_classes)

    def __init_model__(self, input_img_shape, 
                             input_features_shape, 
                             num_classes,
                             use_batch_norm=True,
                             num_filters_last=32):
        #! first part
        inputs_img = Input(shape=input_img_shape, name='img_inputs')
        #* layer 1
        conv_1 = Conv2D(32, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(inputs_img)
        pool_1 = MaxPool2D()(conv_1)

        #* layer 2
        conv_21 = Conv2D(64, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(pool_1)
        conv_22 =  Conv2D(64, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(conv_21)
        norm_2 = BatchNormalization()(conv_22)
        pool_2 = MaxPool2D()(norm_2)

        pool_2 = Dropout(0.5)(pool_2)
        
        #* layer 3
        conv_31 = Conv2D(64, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(pool_2)
        conv_32 =  Conv2D(64, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(conv_31)
        norm_3 = BatchNormalization()(conv_32)
        pool_3 = MaxPool2D()(norm_3)
        
        #* layer 4
        conv_41 = Conv2D(64, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(pool_3)
        conv_42 =  Conv2D(64, (3, 3), padding = "same", 
                                activation='relu',
                                kernel_initializer='he_normal')(conv_41)
        norm_4 = BatchNormalization()(conv_42)
        pool_4 = MaxPool2D()(norm_4)

        pool_4 = Dropout(0.5)(pool_4)

        flatten_img = Flatten()(pool_4)
        dence_im_1 = Dense(1024, activation = "relu")(flatten_img)
        dence_im_1 = Dropout(0.5)(dence_im_1)

        dence_img_last = Dense(num_filters_last, 
                                activation = "relu", 
                                name='img_last_layer')(dence_im_1)
        images = Model(inputs=inputs_img, outputs=dence_img_last)

        #! second part
        inputs_features = Input(shape=input_features_shape, name='feature_inputs')
        flatten_feat = Flatten()(inputs_features)
        dence_f_1 = Dense(1024, activation = "relu")(flatten_feat)
        dence_f_1 = Dropout(0.5)(dence_f_1)
        dence_f_2 = Dense(1024, activation = "relu")(dence_f_1)
        dence_f_2 = Dropout(0.5)(dence_f_2)

        dence_features_last = Dense(num_filters_last,
                                activation = "relu", 
                                name='features_last_layer')(dence_f_2)
        features = Model(inputs=inputs_features, outputs=dence_features_last)

        #! combine two outputs
        combined = Concatenate()([images.output, features.output])
        dence_combined = Dense(1024, activation = "relu")(combined)
        dence_combined = Dropout(0.5)(dence_combined)
        model_outputs = Dense(num_classes, activation = "softmax")(dence_combined)

        self.__model = Model(inputs=[images.input, features.input], outputs=model_outputs)


    def train(self, images, features, labels, 
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

    


def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	return coords

def convert_landmarks(rect, shape):
    x, y, w, h = rect_to_bb(rect)
    cords = shape_to_np(shape, dtype='float32')
    cords -= [x, y]
    cords = 2*cords / [w, h] - 1    # normalize all landmarks
    return cords

def load_data(csv_filename, label_map, new_size=None, include_images=False):
    df = pd.read_csv(csv_filename)
    #Model = Facial_Feature_Net()
    #Model.load_model("models\\76_facial_model.json")
    #Model.load_weights("models\\76_facial_model.h5")
    x_data, y_data = [], []
    images = []
    for index, row in df.iterrows():
        landmarks = np.fromstring(row['features'], 'float32', sep=' ').reshape(68, 2)
        x_data.append(landmarks)
        y_data.append(label_map[row['label']])
        if include_images:
            im = load_img(row['file'], True, None)
            im = im[row['y0']:row['y1'], row['x0']:row['x1']]
            if new_size is not None:
                im = cv2.resize(im, new_size, interpolation = cv2.INTER_AREA)
            images.append(im)
    #x_data, y_data = shuffle(x_data, y_data, random_state=42)
    return  np.array(x_data),  np.array(y_data), np.array(images)

def load_dataset(csv_filename, label_map):
    df = pd.read_csv(csv_filename)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_detection\\shape_predictor_68_face_landmarks.dat')
    x_data, y_data = [], []
    features = []
    for index, row in df.iterrows():
        img = cv2.imread(row['file'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rects = detector(img, 1)
        for k, rect in enumerate(rects):
            shape = predictor(img, rect)
            landmarks = convert_landmarks(rect, shape)
        x_data.append(landmarks)
        y_data.append(label_map[row['label']])
        features.append(' '.join(str(item) for innerlist in landmarks for item in innerlist))
    df = df.assign(features=features)
    df.to_csv(csv_filename)
    x_data, y_data = shuffle(x_data, y_data, random_state=random.seed())
    return  np.array(x_data),  np.array(y_data)

def classify_emotions_with_features(csv_filename, n_epochs=100, batch_size=32, load = False):
    x_data, y_data, _ = load_data(csv_filename,
                    label_map={'neutral' : 0, 'anger' : 1, 'disgust' : 2, 'fear':3, 'happy':4, 'sadness':5, 'surprise':6})
    #x_data = np.reshape(x_data, (-1, 68, 2, 1))
    n_classes = np.unique(y_data).shape[0]
    y_data = np_utils.to_categorical(y_data, n_classes)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)

    if load:
        with open('models\\dlib_facial.json', 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights('models\\dlib_facial.h5')
    else:
        model = Sequential()
        #model.add(Conv2D())
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
    # with open('models\\dlib_facial.json', 'w') as json_file:
    #     json_file.write(model_json)
    # model.save_weights('models\\dlib_facial.h5')

    plot_hist(history_callback)
    return model



def classify_emotions_with_features_combined_model(csv_filename, new_size, n_epochs=100, batch_size=32, load=False, model_id=''):
    features, labels, images = load_data(csv_filename, include_images=True, new_size=new_size,
                    label_map={'neutral' : 0, 'anger' : 1, 'disgust' : 2, 'fear':3, 'happy':4, 'sadness':5, 'surprise':6})
    #x_data = np.reshape(x_data, (-1, 68, 2, 1))
    feature_shape = features.shape[1:]
    images_shape = (new_size[0], new_size[1], 1)
    images = np.reshape(images, (-1, new_size[0], new_size[1], 1))
    n_classes = np.unique(labels).shape[0]
    labels = np_utils.to_categorical(labels, n_classes)
    if load:
        model = Combined_Facial_Net(json_filename='models\\' + model_id + 'model.json', 
                                    h5_filename='models\\' + model_id + 'model.h5')
    else:
        model = Combined_Facial_Net(input_img_shape = images_shape, 
                                    input_features_shape = feature_shape, 
                                    num_classes = n_classes)
    history = model.train(images, features, labels, 
                        optim='adam',
                        n_epochs=n_epochs, 
                        batch_size=batch_size, 
                        save_best_to=None)
    
    model.save_weights('models\\' + model_id + 'model.h5')
    model.save_model('models\\' + model_id + 'model.json')
    
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

    return model


if __name__=='__main__':
    #model = classify_emotions_with_features('data\\dataset.csv', batch_size=512, n_epochs=300, load=True)
    classify_emotions_with_features_combined_model('data\\dataset.csv', 
                                    batch_size=32, new_size=(96, 96), 
                                    n_epochs=50, model_id='facial_comb_')
    plt.show()
    