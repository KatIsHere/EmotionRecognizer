import dlib
import pandas as pd
import os
import numpy as np
import random
import cv2
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

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



def plot_density(csv_filename, label_map = {'anger':0, 'surprise':5, 'disgust':1, 'fear':2, 'neutral':6, 'happiness':3, 'sadness':4, 
                    'ANGER':0, 'SURPRISE':5, 'DISGUST':1, 'FEAR':2, 'NEUTRAL':6, 'HAPPINESS':3, 'SADNESS':4}):
    """
        Plots class density
    """
    df = pd.read_csv(csv_filename)
    c = np.zeros(7)
    for it, row in df.iterrows():
        if row['emotion'] in label_map:
            c[label_map[row['emotion']]] += 1
    plt.plot([0, 1, 2, 3, 4, 5, 6], c, 'o')
    #n, bins, patches = plt.hist(c, 7, facecolor='blue', alpha=0.5)
    plt.show()


def plot_density_v2(csv_filename):
    """
        Plots class density
    """
    df = pd.read_csv(csv_filename)
    c = np.zeros(7)
    for it, row in df.iterrows():
        c[int(row['label'])] += 1
    plt.plot([0, 1, 2, 3, 4, 5, 6], c, 'o')
    #n, bins, patches = plt.hist(c, 7, facecolor='blue', alpha=0.5)
    plt.show()

    
def load_img(im_path, greyscale=False, resize=False, new_size=None):
    """
        Loads an image

        input: 
            * im_path - path to the image
            * greyscale - if true, loads image in greyscae format
            * resize - if true resizes image to new_size
        output:
            * numpy array representing image
    """
    flag = cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR 
    img = cv2.imread(im_path, flag)

    if resize:
        assert len(new_size) == 2
        img = cv2.resize(img, new_size, interpolation = cv2.INTER_AREA)

    return img