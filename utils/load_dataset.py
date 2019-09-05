import glob
import cv2
import numpy as np
import os
import re
import pandas as pd
import dlib
from sklearn.utils import shuffle
import random
import sys, os
from pathlib import Path
ROOT_DIR = Path(__file__).parents[1]
if __name__ == '__main__':
    from utils import load_img, convert_landmarks
else:
    sys.path.append(os.path.abspath(ROOT_DIR))
    from utils.utils import load_img, convert_landmarks

def load_dataset_and_cut_out_faces(csv_filename, label_map, greyscale=True, new_size = None):
    df = pd.read_csv(csv_filename)
    x_data, y_data = [], []
    for index, row in df.iterrows():
        im = load_img(row['file'], greyscale, None)
        if greyscale:
            im = im[row['y0']:row['y1'], row['x0']:row['x1']]
        else:
            im = im[row['y0']:row['y1'], row['x0']:row['x1'], :]
        if new_size is not None:
            im = cv2.resize(im, new_size, interpolation = cv2.INTER_AREA)
        if row['label'] in label_map:
            x_data.append(im)
            y_data.append(label_map[row['label']])
    x_data, y_data = shuffle(x_data, y_data, random_state=42)
    return x_data, y_data


def load_dataset_with_facial_features(csv_filename, label_map, new_size=None, include_images=False, include_augmented=True):
    df = pd.read_csv(csv_filename)
    x_data, y_data = [], []
    images = []
    for index, row in df.iterrows():
        landmarks = np.fromstring(row['features'], 'float32', sep=' ')
        if landmarks.shape[0] == 68*2:
            landmarks = np.reshape(landmarks, (68, 2))
            x_data.append(landmarks)
            y_data.append(label_map[row['label']])

            if include_augmented and type(row['random_augmentation']) is str:
                landmarks_augm = np.fromstring(row['random_augmentation'], 'float32', sep=' ')
                x_data.append(np.reshape(landmarks_augm, (68, 2)))
                y_data.append(label_map[row['label']])

            if include_images:
                im = load_img(row['file'], True, None)
                im = im[row['y0']:row['y1'], row['x0']:row['x1']]
                if new_size is not None:
                    im = cv2.resize(im, new_size, interpolation = cv2.INTER_AREA)
                images.append(im)
                if include_augmented and type(row['random_augmentation']) is str:
                    images.append(im)
    #x_data, y_data = shuffle(x_data, y_data, random_state=42)
    return  np.array(x_data),  np.array(y_data), np.array(images)


def load_dataset_no_face(csv_filename, label_map, new_size = None, greyscale=False):
    augmented_em = ['anger', 'sadness']
    df = pd.read_csv(csv_filename)
    x_data, y_data = [], []
    for index, row in df.iterrows():
        if row['label'] in label_map:
            im = np.fromstring(row['pixels'], sep=' ').astype('uint8')
            #im = np.array([int(x) for x in row['pixels'].split(' ')]).astype('uint8')
            im = np.resize(im, (48, 48))
            if new_size is not None:
                im = cv2.resize(im, new_size, cv2.INTER_AREA)
            if not greyscale:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            x_data.append(im)
            y_data.append(label_map[row['label']])
            if row['label'] in augmented_em:  # augment data that is not represented enough
                x_data.append(cv2.flip(im, 0))
                y_data.append(label_map[row['label']])
    x_data, y_data = shuffle(x_data, y_data, random_state=42)
    return x_data, y_data


def load_dataset_no_face_custom(csv_filename, data_path, new_size = None, greyscale=False, 
                    label_map = {'anger':0, 'surprise':5,  'fear':2, 'neutral':6, 'happiness':3, 'sadness':4, 
                    'ANGER':0, 'SURPRISE':5, 'DISGUST':1, 'FEAR':2, 'NEUTRAL':6, 'HAPPINESS':3, 'SADNESS':4}):
    df = pd.read_csv(csv_filename)
    x_data, y_data = [], []
    for index, row in df.iterrows():
        if row['emotion'] in label_map:
            path = data_path + row['image']
            im = load_img(path, greyscale, None)
            if im is None:
                continue
            if new_size is not None:
                im = cv2.resize(im, new_size, cv2.INTER_AREA)
            x_data.append(im)
            y_data.append(label_map[(row['emotion'])])
    x_data, y_data = shuffle(x_data, y_data, random_state=42)
    return x_data, y_data
    

def load_facial_dataset_for_autoencoder(img_dir, csv_filename, greyscale=True, new_size=None):
    # bbox_x0,bbox_y0,bbox_x1,bbox_y1
    df = pd.read_csv(csv_filename)
    x_data, y_data = [], []
    for index, row in df.iterrows():
        im = load_img(img_dir + row['name'] + '.jpg', greyscale, None)
        if greyscale:
            im = im[row['bbox_y0']:row['bbox_y1'], row['bbox_x0']:row['bbox_x1']]
        else:
            im = im[row['bbox_y0']:row['bbox_y1'], row['bbox_x0']:row['bbox_x1'], :]
        if new_size is not None:
            im = cv2.resize(im, new_size, interpolation = cv2.INTER_AREA)
        x_data.append(im)
        coords = row.iloc[3:-4].values
        w_org = new_size[1] / (row['bbox_x1'] -  row['bbox_x0'])
        h_org = new_size[0] / (row['bbox_y1'] -  row['bbox_y0'])
        coords = np.reshape(np.array(coords), (76, 2))
        coords = (coords - [row['bbox_x0'], row['bbox_y0']]) * [w_org, h_org]
        coords = coords.astype('int')
        coords = np.clip(coords, 0, [new_size[1] - 1, new_size[0] - 1])
        mask = np.zeros((new_size), dtype='float32')
        mask[[point for point in coords]] = 1
        y_data.append(mask)
    x_data, y_data = shuffle(x_data, y_data, random_state=42)
    return x_data, y_data
