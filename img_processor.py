
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
import matplotlib.pyplot as plt

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


def predict_and_vizualize(csv_filename):
    model = Emotion_Net()
    model.load_model("models\\combined_model.json")
    model.load_weights("models\\combined_model.h5")
    df = pd.read_csv(csv_filename)
    smp = df.sample(5, random_state=42)
    for index, row in smp.iterrows():
        img = load_img(row['file'], False, None)
        img = detect_and_classify(img, model)
        cv2.imshow('img %d'.format(index), img)
        cv2.waitKey(0)


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


def normalize_data(x_data, y_data, im_rows, im_cols, channels=1):
        # normalizing and preparing data
        x_data = np.array(x_data)
        y_data = np.array(y_data, dtype='int32')
        x_data = x_data.astype('float32')
        n_classes = np.unique(y_data).shape[0]
        x_data /= 255.0   
        y_data = np_utils.to_categorical(y_data, n_classes)
        x_data = x_data.reshape(x_data.shape[0], im_rows, im_cols, channels)
        return x_data, y_data, n_classes


def train_keras_model(dataset_csv, 
                        im_shape, 
                        model_folder = 'models\\', 
                        augment=False,
                        model_id='combined_', 
                        load_weights=True, 
                        plot_metrix=False, 
                        channels=1,
                        epocs=50,
                        arc=0):
        """Trains a model based on kanade database and saves it's structure and weights"""
        gray = True if channels==1 else False
        x_data, y_data = load_dataset_csv(dataset_csv, new_size=im_shape, greyscale=gray)
        im_rows, im_cols = im_shape
        input_shape = (im_rows, im_cols, channels)
        x_data, y_data, n_classes = normalize_data(x_data, y_data, im_rows, im_cols, channels)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        Model = Emotion_Net()

        if load_weights:
                Model.load_model(model_folder + model_id + "model.json")
                Model.load_weights(model_folder + model_id + "model.h5")
        else:
                Model.init_model(input_shape, n_classes, arc=arc)
        if augment:
                history_call = Model.augment_and_train(x_train, y_train, x_test, y_test, save_best_to=model_folder + model_id + "model.hdf5", \
                        batch_size=32, n_epochs=epocs, optim=Adam(lr=0.0001))
        else:
                history_call = Model.train(x_train, y_train, x_test, y_test, save_best_to=model_folder + model_id + "model.hdf5", \
                        batch_size=32, n_epochs=epocs, optim=Adam(lr=0.0001))
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
                        batch_size=32, n_epochs=20, optim=Adam(lr=0.0005))
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


# ? Short term tasks
# TODO: visualize
# TODO: need to create separate validation database
if __name__ == "__main__":
    random.seed()
    #predict_and_vizualize('data\\dataset.csv')
    (im_rows, im_cols) = (256, 256)
    print("KANADE RESNET")
    n_classes = train_keras_model('data\\dataset_kanade.csv', im_shape = (im_rows, im_cols), epocs=30,
                       load_weights=False, model_id='kanade_resnet_', plot_metrix=True, channels=3, arc=2)    
    print("CUSTOM RESNET")
    n_classes = train_keras_model('data\\dataset.csv', im_shape = (im_rows, im_cols), epocs=30,
                       load_weights=False, model_id='combined_resnet_', plot_metrix=True, channels=3, arc=2)
    print("CUSTOM VGG16")
    n_classes = train_keras_model('data\\dataset.csv', im_shape = (im_rows, im_cols), epocs=40,
                       load_weights=False, model_id='combined_vgg16_', plot_metrix=True, channels=3, arc=1)
    #n_classes = 7
    print("TESTING KANADE")
    print("Testing on jaffe")
    validate_on_database("data\\dataset_jaffe.csv", "models\\kanade_vgg16_model", 
                n_classes, im_shape = (im_rows, im_cols), channels=3)
    print("Testing on facesdb")
    validate_on_database("data\\dataset_facesdb.csv", "models\\kanade_vgg16_model", 
                n_classes, im_shape = (im_rows, im_cols), channels=3)
    
    print("TESTING COMBINED")
    print("Testing on jaffe")
    validate_on_database("data\\dataset_jaffe.csv", "models\\combined_vgg16_model", 
                n_classes, im_shape = (im_rows, im_cols), channels=3)
    print("Testing on facesdb")
    validate_on_database("data\\dataset_facesdb.csv", "models\\combined_vgg16_model", 
                n_classes, im_shape = (im_rows, im_cols), channels=3)

    plt.show()