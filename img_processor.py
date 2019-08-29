
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from load_pics import load_dataset_csv, load_img, load_dataset_no_face, load_dataset_no_face_custom
from face_detector import faces_from_database_dnn, find_faces_dnn
from keras.optimizers import SGD, Adam
from keras_model import Emotion_Net
from keras.utils import np_utils
import numpy as np
import random
import pandas as pd
import cv2
from prepare_dataset import DataPreprocessor

label_map = {'neutral' : 0, 'anger' : 1, 'disgust' : 2, 'fear':3, 'happy':4, 'sadness':5, 'surprise':6}

def detect_and_classify(img, model, conf_threshold = 0.97, new_size=(200, 200), channels=1,
                lbl_map = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}):
    """
        0 ----- 'anger', 
        1 ----- 'disgust'
        2 ----- 'fear'
        3 ----- 'happiness'
        4 ----- 'sadness'
        5 ----- 'surprise'
        6 ----- 'neutral'
    """
    detections = find_faces_dnn(img)
    (h, w) = img.shape[:2]
    faces = []
    bboxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence >= conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            if (box <= [w, h, w, h]).all() and (box >= [0., 0., 0., 0.]).all():
                (startX, startY, endX, endY) = box.astype("int")
                face = img[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                if new_size is not None:
                        assert len(new_size) == 2
                        face = cv2.resize(face, new_size, interpolation = cv2.INTER_AREA)
                faces.append(np.resize(face, (face.shape[0], face.shape[1], channels)))
                bboxes.append([(startX, startY), (endX, endY)])
    faces = np.array(faces, dtype='float32')
    if faces.shape[0] == 0:
            return img
    faces = faces / 123.0 - 1
    expressions = model.predict(faces)
    for i in range(len(bboxes)):
        img = cv2.rectangle(img, bboxes[i][0], bboxes[i][1], (0, 0, 255), 2)
        img = cv2.putText(img, lbl_map[np.argmax(expressions[i])], (bboxes[i][0][0], bboxes[i][0][1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img


def test_model(model_id, model_folder = "models\\"):
        model = Emotion_Net()
        model.load_model(model_folder + model_id + "model.json")
        model.load_weights(model_folder + model_id + "model.h5")
        test_imgs = ['data\\test_img.png', 'data\\test_img2.jpg', 'data\\test_img3.jpg', 'data\\test_img4.jpg']
        for img_name in test_imgs:
                img = load_img(img_name, False, None)
                img = detect_and_classify(img, model)
                cv2.imshow(img_name, img)
                cv2.waitKey(0)        

def open_test_im(csv_filename):
        df = pd.read_csv(csv_filename)
        smp = df.sample(5, random_state=42)
        for index, row in smp.iterrows():
                im = np.array([int(x) for x in row['pixels'].split(' ')])
                img = np.reshape(im, (48, 48, 1))
                img = img.astype('float')
                img /= 255.0
                img = cv2.resize(img, (96, 96), cv2.INTER_AREA)
                cv2.imshow('img 1', img)
                cv2.waitKey(0)

def predict_and_vizualize(csv_filename):
    model = Emotion_Net()
    model.load_model("models\\ins_resnet_v3_newdata_model.json")
    model.load_weights("models\\ins_resnet_v3_newdata_model.h5")
    df = pd.read_csv(csv_filename)
    smp = df.sample(5, random_state=42)
    for index, row in smp.iterrows():
        img = load_img(row['file'], False, None)
        img = detect_and_classify(img, model)
        cv2.imshow('img %d'.format(index), img)
        cv2.waitKey(0)

def normalize_data(x_data, y_data, im_rows, im_cols, channels=1):
        # normalizing and preparing data
        x_data = np.array(x_data, dtype='float32')
        y_data = np.array(y_data, dtype='int32')
        n_classes = np.unique(y_data).shape[0]
        x_data = (x_data - 123.0) / 123.0
        y_data = np_utils.to_categorical(y_data, n_classes)
        x_data = x_data.reshape(x_data.shape[0], im_rows, im_cols, channels)
        return x_data, y_data, n_classes

def validate_on_database(csv_filename, model_filename, n_classes, im_shape, channels=1):
        gray = True if channels==1 else False
        model = Emotion_Net()
        model.load_model(model_filename +".json")
        model.load_weights(model_filename + ".h5")
        x_val, y_val = load_dataset_csv(csv_filename, new_size=im_shape, greyscale=gray)
        x_val = np.array(x_val)
        y_val = np.array(y_val, dtype='int32')
        x_val = x_val.astype('float32')
        x_val /= 255.0   
        y_val = np_utils.to_categorical(y_val, n_classes)
        x_val = x_val.reshape(x_val.shape[0], im_shape[0], im_shape[1], channels)

        model.evaluate_accur(x_val, y_val)


def train_keras_model(dataset_csv, 
                        im_shape, 
                        model_folder = 'models\\', 
                        augment=False,
                        detect_face=True,
                        model_id='', 
                        save_model_id = '',
                        load_weights=True, 
                        plot_metrix=False, 
                        channels=1,
                        epocs=50,
                        batch_size=32,
                        arc=0):
        """Trains a model based on kanade database and saves it's structure and weights"""
        gray = True if channels==1 else False
        im_rows, im_cols = im_shape
        if detect_face:
                x_data, y_data = load_dataset_csv(dataset_csv, label_map, new_size=(im_rows, im_cols), greyscale=gray)
        else:
                #x_data, y_data = load_dataset_no_face(dataset_csv, new_size=(im_rows, im_cols), greyscale=gray)
                x_data, y_data = load_dataset_no_face_custom(dataset_csv, new_size=(im_rows, im_cols), greyscale=gray)
        input_shape = (im_rows, im_cols, channels)
        x_data, y_data, n_classes = normalize_data(x_data, y_data, im_rows, im_cols, channels)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=random.seed())

        Model = Emotion_Net()

        if load_weights:
                Model.load_model(model_folder + model_id + "model.json")
                Model.load_weights(model_folder + model_id + "model.h5")
        else:
                Model.init_model(input_shape, n_classes, arc=arc)
        if augment:
                history_call = Model.augment_and_train(x_train, y_train, x_test, y_test, 
                        save_best_to=model_folder + save_model_id + "model.h5", \
                        batch_size=batch_size, n_epochs=epocs, optim=Adam(lr=0.0001))
        else:
                history_call = Model.train(x_train, y_train, x_test, y_test, 
                        save_best_to=model_folder + save_model_id + "model.h5", \
                        batch_size=batch_size, n_epochs=epocs, optim=Adam(lr=0.0001))
        if plot_metrix:
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

        Model.evaluate_accur(x_test, y_test)
        Model.save_model(model_folder + save_model_id + "model.json")
        #Model.save_weights(model_folder + save_model_id + "model.h5")
        return n_classes


# validating on a completly different dataset
def train_keras_model_with_validation(dataset_csv, validation_csv, 
                        im_shape, 
                        model_folder = 'models\\', 
                        model_id='combined_', 
                        load_weights=True, 
                        plot_metrix=False, 
                        channels=1):
        """Trains a model based on kanade database and saves it's structure and weights"""
        im_rows, im_cols = im_shape
        gray = True if channels==1 else False
        x_train, y_train = load_dataset_csv(dataset_csv, new_size=im_shape, greyscale=gray)
        x_test, y_test = load_dataset_csv(validation_csv, new_size=im_shape, greyscale=gray)

        input_shape = (im_rows, im_cols, channels)
        x_train, y_train, n_classes = normalize_data(x_train, y_train, im_rows, im_cols)
        x_test, y_test, _ = normalize_data(x_test, y_test, im_rows, im_cols)        

        Model = Emotion_Net()

        if load_weights:
                Model.load_model(model_folder + "kanade_model.json")
                Model.load_weights(model_folder + "kanade_model.h5")
        else:
                Model.init_model(input_shape, n_classes)

        history_call = Model.train(x_train, y_train, x_test, y_test, save_best_to=model_folder + model_id + "model.hdf5", \
                        batch_size=16, n_epochs=20, optim=Adam(lr=0.0005))
        if plot_metrix:
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

        Model.evaluate_accur(x_test, y_test)
        Model.save_model(model_folder + model_id + "model.json")
        Model.save_weights(model_folder + model_id + "model.h5")
        return n_classes



if __name__ == "__main__":
    random.seed()
    
    #open_test_im('data\\fer2013.csv')

    predict_and_vizualize('data\\dataset.csv')
    
    (im_rows, im_cols) = (96, 96)
    channels = 3
    n_classes = 7
#     n_classes = train_keras_model('data\\legend.csv', 
#                                 im_shape=(im_rows, im_cols), 
#                                 channels=channels, 
#                                 epocs=5, 
#                                 batch_size=32,
#                                 augment=False, 
#                                 detect_face=False,
#                                 load_weights=True, 
#                                 model_id='ins_resnet_v2_newdata_', 
#                                 save_model_id='ins_resnet_v3_newdata_',
#                                 plot_metrix=True, 
#                                 arc=3)    
#     print("TESTING KANADE")
#     print("Testing on jaffe")
#     validate_on_database("data\\dataset_jaffe.csv", "models\\kanade_mobnet_model", 
#                 n_classes, im_shape = (im_rows, im_cols), channels=channels)
#     print("Testing on facesdb")
#     validate_on_database("data\\dataset_facesdb.csv", "models\\kanade_mobnet_model", 
#                 n_classes, im_shape = (im_rows, im_cols), channels=channels)
    
    print("TESTING COMBINED")
    validate_on_database("data\\dataset.csv", "models\\ins_resnet_v2_newdata_model", 
                n_classes, im_shape = (im_rows, im_cols), channels=channels)
#     print("Testing on facesdb")
#     validate_on_database("data\\dataset_facesdb.csv", "models\\combined_mobnet_model", 
#                 n_classes, im_shape = (im_rows, im_cols), channels=channels)
    plt.show()

#     test_model('combined_arc3_')
# TODO: check mod on test