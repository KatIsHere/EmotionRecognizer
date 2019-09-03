import glob
import cv2
import numpy as np
import os
import re
import pandas as pd
from face_detector import cut_out_faces_dnn, init_model_dnn
from sklearn.utils import shuffle


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


def load_dataset_csv(csv_filename, label_map, greyscale=True, new_size = None):
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



def load_dataset_no_face(csv_filename, label_map, new_size = None, greyscale=False):
    augmented_em = ['anger', 'sadness']
    df = pd.read_csv(csv_filename)
    x_data, y_data = [], []
    for index, row in df.iterrows():
        if row['emotion'] in label_map:
            im = np.fromstring(row['pixels'], sep=' ').astype('uint8')
            #im = np.array([int(x) for x in row['pixels'].split(' ')]).astype('uint8')
            im = np.resize(im, (48, 48))
            if new_size is not None:
                im = cv2.resize(im, new_size, cv2.INTER_AREA)
            if not greyscale:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            x_data.append(im)
            y_data.append(label_map[row['emotion']])
            if row['emotion'] in augmented_em:  # augment data that is not represented enough
                x_data.append(cv2.flip(im, 0))
                y_data.append(label_map[row['emotion']])
    x_data, y_data = shuffle(x_data, y_data, random_state=42)
    return x_data, y_data


def load_dataset_no_face_custom(csv_filename, new_size = None, greyscale=False, 
                    label_map = {'anger':0, 'surprise':5,  'fear':2, 'neutral':6, 'happiness':3, 'sadness':4, 
                    'ANGER':0, 'SURPRISE':5, 'DISGUST':1, 'FEAR':2, 'NEUTRAL':6, 'HAPPINESS':3, 'SADNESS':4}):
    df = pd.read_csv(csv_filename)
    x_data, y_data = [], []
    for index, row in df.iterrows():
        if row['emotion'] in label_map:
            path = 'data\\images\\' + row['image']
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


def load_facial_data_kadle_cvs(csv_filename, cols=None):
    df = pd.read_csv(csv_filename)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    df = df.dropna() 
    if cols:
        df = df[list(cols) + ['Image']]
    df = df.dropna()  
    x_data = np.vstack(df['Image'].values)
    y_data = df[df.columns[:-1]].values
    y_data = (y_data - 48) / 48     # normalizing labels
    x_data, y_data = shuffle(x_data, y_data, random_state=42)
    return x_data, y_data


def load_facial_data_kadle2d(csv_filename, cols=None):
    x_data, y_data = load_facial_data_kadle_cvs(csv_filename, cols)
    x_data = x_data.reshape(-1, 96, 96, 1)
    return x_data, y_data


def load_jaffe(folder_dir, f_format, greyscale=False,
                lbl_dict = {'HA' : 0, 'SA' : 1, 'SU' : 2, 'AN' : 3, 'DI' : 4, 'FE' : 5, 'NE' : 6}):
    """ Loads jaffe dataset

        input: 
            * folder_dir - path to database
            * f_format - images format
            * greyscale - if True load images in greyscale;   
            * lbl_dict - labels interpritator

        output: 
            * an array of images and an array of image labes

    Emotion formating(could be tuned though):
      HAP SAD SUR ANG DIS FEA NET --> 0 1 2 3 4 5 6 """
    assert isinstance(folder_dir, str) and isinstance(f_format, str) 
    f_list = glob.glob(folder_dir + "/*." + f_format)
    data = np.array([load_img(fname, greyscale=greyscale) for fname in f_list])
    strt_ind = f_list[0].find('jaffe') + 9
    labels = np.array([lbl_dict[fname[strt_ind : strt_ind + 2]] for fname in f_list])
    return data, labels


# TODO: give all images to cv.bolb?
def load_kanade(img_folder_dir, label_folder_dir, new_im_size = None, file_format='png', load_grey=True):
    """ Loads kanade dataset

        in: 
            * img_folder_dir - path to database (images); 
            * label_folder_dir - path to image labels;   
            * new_im_size - the new size of loaded images
            * file_format - image format;                 
            * load_grey - if True load images in greyscale;   

        out: 
            * an array of images and an array of image labes

        Emotion labeling: 0 : neutral, 1 : anger, 2 : contempt, 
            3 : disgust, 4 : fear, 5 : happy, 6 : sadness, 7 : surprise 
    """
    labeles = []
    pics = []
    net = init_model_dnn()
    # Firstly we find out if image is labeled and if yes we load it to the database
    for root, dirs, files in os.walk(label_folder_dir):
        for file in files:
            if '.txt' in file:
                labeled_pics_path = os.path.join(root, file)
                #? images will be in a similare directory, 
                #? except file format and root directory will be different 
                pic_file = file.replace('_emotion.txt', "." + file_format)
                pic_root = root.replace(label_folder_dir, img_folder_dir)
                pics_path = os.path.join(pic_root, pic_file)
                im = load_img(pics_path, False, None)

                # we fiind face on the image and save it to the database
                faces = cut_out_faces_dnn(im, net=net, new_size=new_im_size, to_greyscale=True, conf_threshold=0.97)
                
                if faces.shape[0] != 0:
                    pics.append(faces[0])
                    with open(labeled_pics_path, 'r') as lbl:
                        labeles.append(float(lbl.read()[:-1]) - 1.0)
    return pics, np.array(labeles).astype('int32')
    