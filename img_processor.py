
from sklearn.model_selection import train_test_split
from load_pics import load_jaffe, load_kanade
from face_detector import faces_from_database_dnn
from keras.optimizers import SGD, Adam
from keras_model import Emotion_Net
from keras.utils import np_utils
import numpy as np
from prepare_dataset import DataPreprocessor



# TODO: cut out last 3 frames instead of 1
# TODO: add jaffa data to main data(CHANGE LABELS)
# TODO: load weights before running train again
# TODO: see if torch models can be optimized
# TODO: visualize
# TODO: augment data (in the future)
def train_kanade_model(model_folder = 'models\\'):
    """Trains a model based on kanade database and saves it's structure and weights"""
    model_id = 'kanade_'
    #(im_rows, im_cols) = (490//2, 640//2)
    (im_rows, im_cols) = (300, 300)
    x_data, y_data = [], []
    #x_data, y_data = load_jaffe("project\\jaffe", 'tiff', lbl_dict = {'NE' : 0, 'AN' : 1, 'DI' : 3, 'FE' : 4, 'HA' : 5, 'SA' : 6, 'SU' : 7 } )
    #x_data, y_data = load_kanade("kanade\\cohn-kanade-images\\", "kanade\\emotion\\", new_im_size=(im_rows, im_cols), load_grey=False)
    x_data = np.array(x_data)
    n_classes = np.unique(y_data).shape[0]
    #im_rows, im_cols = x_data.shape[1], x_data.shape[2]

    input_shape = (im_rows, im_cols, 1)

    # normalizing and preparing data
    x_data = x_data.astype('float32')
    x_data /= 255.0   
    y_data = np_utils.to_categorical(y_data, n_classes)
    x_data = x_data.reshape(x_data.shape[0], im_rows, im_cols, 1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    Model = Emotion_Net(input_shape, n_classes)
    Model.train(x_train, y_train, x_test, y_test, save_best_to=model_folder + model_id + "model.hdf5", \
                batch_size=32, n_epochs=150, optim=Adam(lr=0.0005))
    Model.evaluate_accur(x_test, y_test)
    Model.save_model(model_folder + model_id + "model.json")
    Model.save_weights(model_folder + model_id + "model.h5")


#train_kanade_model()
