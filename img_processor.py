
from sklearn.model_selection import train_test_split
from load_pics import load_dataset_csv
from face_detector import faces_from_database_dnn
from keras.optimizers import SGD, Adam
from keras_model import Emotion_Net
from keras.utils import np_utils
import numpy as np
from prepare_dataset import DataPreprocessor

def train_kanade_model(model_folder = 'models\\'):
    """Trains a model based on kanade database and saves it's structure and weights"""
    #model_id = 'combined_'
    model_id = 'kanade_'
    (im_rows, im_cols) = (300, 300)
    x_data, y_data = [], []

    #x_data, y_data = load_jaffe("project\\jaffe", 'tiff', lbl_dict = {'NE' : 0, 'AN' : 1, 'DI' : 3, 'FE' : 4, 'HA' : 5, 'SA' : 6, 'SU' : 7 } )
    #x_data, y_data = load_kanade("kanade\\cohn-kanade-images\\", "kanade\\emotion\\", new_im_size=(im_rows, im_cols), load_grey=False)

    x_data, y_data = load_dataset_csv('data\\dataset_kanade.csv', new_size=(im_rows, im_cols))

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

    Model = Emotion_Net(input_shape, n_classes)
    #Model.load_weights(model_folder + "kanade_model.h5")
    Model.train(x_train, y_train, x_test, y_test, save_best_to=model_folder + model_id + "model.hdf5", \
                batch_size=32, n_epochs=100, optim=Adam(lr=0.001))
    Model.evaluate_accur(x_test, y_test)
    Model.save_model(model_folder + model_id + "model.json")
    Model.save_weights(model_folder + model_id + "model.h5")



#? Long term plans:
# TODO: see if torch models can be optimized
# TODO: visualize
# TODO: augment data (in the future)
if __name__ == "__main__":
    train_kanade_model()
