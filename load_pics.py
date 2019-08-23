import glob
import cv2
import numpy as np
import os
import re
import pandas as pd
from face_detector import cut_out_faces_dnn, init_model_dnn

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


def load_dataset_csv(csv_filename, greyscale=True, new_size = None):
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
        x_data.append(im)
        y_data.append(row['label'])
    return x_data, y_data
    

def load_facial_dataset_csv(img_dir, csv_filename, greyscale=True, new_size=None):
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
        w_org = 1 / (row['bbox_x1'] -  row['bbox_x0'])
        h_org = 1 / (row['bbox_y1'] -  row['bbox_y0'])
        for i in range(0, len(coords) - 1, 2):
            coords[i] = (coords[i] - row['bbox_x0']) * w_org
            coords[i + 1] = (coords[i + 1] - row['bbox_y0']) * h_org
        y_data.append(coords)
    return x_data, y_data


# TODO: rename labels and cut out face before passing down
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
    
    
# TODO: rewrite using tiff package
def load_jaffe_tiff(folder_dir, 
                lbl_dict = {'HA' : 0, 'SA' : 1, 'SU' : 2, 'AN' : 3, 'DI' : 4, 'FE' : 5, 'NE' : 6}):
    pass