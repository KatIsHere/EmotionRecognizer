import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, model_from_json, Model 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D
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

def load_data(csv_filename, label_map, new_size=(96, 96)):
    df = pd.read_csv(csv_filename)
    #Model = Facial_Feature_Net()
    #Model.load_model("models\\76_facial_model.json")
    #Model.load_weights("models\\76_facial_model.h5")
    x_data, y_data = [], []
    for index, row in df.iterrows():
        landmarks = np.fromstring(row['features'], 'float32', sep=' ').reshape(68, 2)
        # im = load_img(row['file'], True, None)
        # im = im[row['y0']:row['y1'], row['x0']:row['x1']]
        # if new_size is not None:
        #     im = cv2.resize(im, new_size, interpolation = cv2.INTER_AREA)
        # im = np.reshape(im, (1, 96, 96, 1))
        # landmarks = Model.predict(im)
        x_data.append(landmarks)
        y_data.append(label_map[row['label']])
    x_data, y_data = shuffle(x_data, y_data, random_state=42)
    return  np.array(x_data),  np.array(y_data)

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
    x_data, y_data = load_data(csv_filename,
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


if __name__=='__main__':
    model = classify_emotions_with_features('data\\dataset.csv', batch_size=512, n_epochs=300, load=True)

    plt.show()
    