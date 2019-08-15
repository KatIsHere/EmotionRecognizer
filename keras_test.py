import tensorflow as tf
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from load_pics import load_jaffe

model_folder = 'C:\\Users\\Cat\\Documents\\Labs\\abto_practice\\project\\models\\'

x_data, y_data = load_jaffe("project\\jaffe", 'tiff')
n_classes = np.unique(y_data).shape[0]
im_rows, im_cols = x_data.shape[1], x_data.shape[2]

input_shape = (im_rows, im_cols, 1)

# normalizing and preparing data
x_data = x_data.astype('float32')
x_data /= 255.0   
y_data = np_utils.to_categorical(y_data, n_classes)
x_data = x_data.reshape(x_data.shape[0], im_rows, im_cols, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=47)

model = Sequential()

Conv2D(32, (7, 7), padding = "same", input_shape = input_shape, activation = 'relu')
Conv2D(64, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu')
BatchNormalization(epsilon=0.0001)
MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same")

#Conv2D(64, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu')
#MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same")

Conv2D(128, (1, 1), padding = "same", input_shape = input_shape, activation = 'relu')
BatchNormalization(epsilon=0.0001)
MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same")

Dropout(0.4)

Conv2D(256, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu')
BatchNormalization(epsilon=0.0001)
MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same")

Conv2D(256, (1, 1), padding = "same", input_shape = input_shape, activation = 'relu')
BatchNormalization(epsilon=0.0001)
MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same")

#Dropout(0.3)

Conv2D(512, (1, 1), padding = "same", input_shape = input_shape, activation = 'relu')
BatchNormalization(epsilon=0.0001)
MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = "same")


# tensor reforming 
model.add(Flatten())
model.add(Dense(1024, activation = "relu"))
model.add(Dropout(0.3))     # reg
model.add(Dense(2048, activation = "relu"))
model.add(Dropout(0.5))     # reg
model.add(Dense(n_classes, activation = 'softmax'))

# optimizing
model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.001), metrics = ["accuracy"])
# learning
model.fit(x_train, y_train, batch_size = 16, callbacks = [ModelCheckpoint( model_folder + "model_v2.hdf5", monitor = "val_acc",      \
            save_best_only = True, save_weights_only = False, mode = "auto")], epochs = 300,    \
            verbose = 1, shuffle = True, validation_data = (x_test, y_test))
# accuracy
score = model.evaluate(x_test, y_test, verbose = 0)
print("Test score :", score[0])
print("Test accuracy :", score[1], "\n")

# saving model to .json file
model_json = model.to_json()
with open(model_folder + "model_v2.json", 'w') as json_file:
    json_file.write(model_json)
# saving weights to .hdf5 file
model.save_weights(model_folder + "model_v2.h5")
print("Model saved\n")