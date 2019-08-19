
from sklearn.model_selection import train_test_split
from load_pics import load_dataset_csv, load_img
from face_detector import faces_from_database_dnn, find_faces_dnn
from keras.optimizers import SGD, Adam
from keras_model import Emotion_Net
from keras.utils import np_utils
import numpy as np
import random
import pandas as pd
import cv2
from prepare_dataset import DataPreprocessor

def detect_and_classify(img, model, conf_threshold = 0.98, new_size=(300, 300)):
    detections = find_faces_dnn(img)
    (h, w) = img.shape[:2]
    faces = []
    bboxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = img[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            if new_size is not None:
                assert len(new_size) == 2
                face = cv2.resize(face, new_size, interpolation = cv2.INTER_AREA)
            faces.append(np.resize(face, (face.shape[0], face.shape[1], 1)))
            bboxes.append([(startX, startY), (endX, endY)])
    
    expressions = model.predict(np.array(faces))
    for bbox in bboxes:
        img = cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 2)
        for espr in range(0, expressions.shape[0]):
            img = cv2.putText(img, str(expressions[espr][0]), bbox[0], 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            img = cv2.putText(img, str(expressions[espr][1]), (bbox[0][0] + 40, bbox[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            img = cv2.putText(img, str(expressions[espr][2]), (bbox[0][0] + 80, bbox[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            img = cv2.putText(img, str(expressions[espr][3]), (bbox[0][0] + 120, bbox[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            img = cv2.putText(img, str(expressions[espr][4]), (bbox[0][0] + 160, bbox[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            img = cv2.putText(img, str(expressions[espr][5]), (bbox[0][0] + 200, bbox[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            img = cv2.putText(img, str(expressions[espr][6]), (bbox[0][0] + 240, bbox[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            img = cv2.putText(img, str(expressions[espr][7]), (bbox[0][0] + 280, bbox[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                

    return img



def predict_and_vizualize(csv_filename, model):
    df = pd.read_csv(csv_filename)
    smp = df.sample(5, random_state=42)
    for index, row in smp.iterrows():
        img = load_img(row['file'], False, None)
        img = detect_and_classify(img, model)
        cv2.imshow('img %d'.format(index), img)
        cv2.waitKey(0)
    

def train_kanade_model(model_folder = 'models\\'):
    """Trains a model based on kanade database and saves it's structure and weights"""
    #model_id = 'combined_'
    model_id = 'combined_'
    (im_rows, im_cols) = (300, 300)
    x_data, y_data = [], []

    #x_data, y_data = load_jaffe("project\\jaffe", 'tiff', lbl_dict = {'NE' : 0, 'AN' : 1, 'DI' : 3, 'FE' : 4, 'HA' : 5, 'SA' : 6, 'SU' : 7 } )
    #x_data, y_data = load_kanade("kanade\\cohn-kanade-images\\", "kanade\\emotion\\", new_im_size=(im_rows, im_cols), load_grey=False)

    x_data, y_data = load_dataset_csv('data\\dataset.csv', new_size=(im_rows, im_cols))

    x_data = np.array(x_data)
    y_data = np.array(y_data, dtype='int32')
    x_data = x_data.astype('float32')
    n_classes = np.unique(y_data).shape[0]
    input_shape = (im_rows, im_cols, 1)

    # normalizing and preparing data
    x_data /= 255.0   
    y_data = np_utils.to_categorical(y_data, n_classes)
    x_data = x_data.reshape(x_data.shape[0], im_rows, im_cols, 1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    Model = Emotion_Net()
    #Model.init_model(input_shape, n_classes)
    Model.load_model(model_folder + "kanade_model.json")
    Model.load_weights(model_folder + "kanade_model.h5")
    Model.train(x_train, y_train, x_test, y_test, save_best_to=model_folder + model_id + "model.hdf5", \
                batch_size=32, n_epochs=50, optim=Adam(lr=0.0005))
    Model.evaluate_accur(x_test, y_test)
    Model.save_model(model_folder + model_id + "model.json")
    Model.save_weights(model_folder + model_id + "model.h5")



#? Long term plans:
# TODO: see if torch models can be optimized
# TODO: visualize
# TODO: augment data (in the future)
if __name__ == "__main__":
    random.seed(9001)
    Model = Emotion_Net()
    Model.load_model("models\\combined_model.json")
    Model.load_weights("models\\combined_model.h5")
    predict_and_vizualize('data\\dataset.csv', Model)
    #train_kanade_model()
