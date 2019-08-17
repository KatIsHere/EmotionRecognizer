import glob
from PIL import Image
import cv2
import numpy as np
import os
import re


def load_jaffe(folder_dir, f_format, 
                lbl_dict = {'HA' : 0, 'SA' : 1, 'SU' : 2, 'AN' : 3, 'DI' : 4, 'FE' : 5, 'NE' : 6}):
    """ Loads jaffe dataset

        in: 
            * folder_dir - path to database
            * f_format - images format

        out: 
            * an array of images and an array of image labes

    Emotion formating(could be tuned though):
      HAP SAD SUR ANG DIS FEA NET --> 0 1 2 3 4 5 6 """
    assert isinstance(folder_dir, str) and isinstance(f_format, str) 
    f_list = glob.glob(folder_dir + "/*." + f_format)
    data = np.array([np.array(Image.open(fname)) for fname in f_list])
    strt_ind = f_list[0].find('jaffe') + 9
    labels = np.array([lbl_dict[fname[strt_ind : strt_ind + 2]] for fname in f_list])
    return data, labels


def load_img(im_path, greyscale=False, resize=False, new_size=None):
    flag = cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_UNCHANGED 
    img = cv2.imread(im_path, flag)

    if resize:
        assert len(new_size) == 2
        img = cv2.resize(img, new_size, interpolation = cv2.INTER_AREA)

    return img


def load_kanade(img_folder_dir, label_folder_dir, new_im_size, file_format='png', load_grey=True):
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
                pics.append(load_img(pics_path, load_grey, True, new_im_size))

                with open(labeled_pics_path, 'r') as lbl:
                    labeles.append(float(lbl.read()[:-1]))
    return np.array(pics), np.array(labeles).astype('int32')
    
    

# TODO: rewrite using tiff package
def load_jaffe_tiff(folder_dir, 
                lbl_dict = {'HA' : 0, 'SA' : 1, 'SU' : 2, 'AN' : 3, 'DI' : 4, 'FE' : 5, 'NE' : 6}):
    pass