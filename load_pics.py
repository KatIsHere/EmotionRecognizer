import glob
from PIL import Image
import numpy as np


def load_jaffe(folder_name, f_format, 
                lbl_dict = {'HA' : 0, 'SA' : 1, 'SU' : 2, 'AN' : 3, 'DI' : 4, 'FE' : 5, 'NE' : 6}):
    """ Loads jaffe dataset
        in: folder_name - path to database
            f_format - images format
        out: an array of images and an array of image labes
    Emotion formating(could be tuned though):
      HAP SAD SUR ANG DIS FEA NET --> 0 1 2 3 4 5 6 """
    assert isinstance(folder_name, str) and isinstance(f_format, str) 
    f_list = glob.glob(folder_name + "/*." + f_format)
    data = np.array([np.array(Image.open(fname)) for fname in f_list])
    strt_ind = f_list[0].find('jaffe') + 9
    labels = np.array([lbl_dict[fname[strt_ind : strt_ind + 2]] for fname in f_list])
    return data, labels

    
# TODO: rewrite using tiff package
def load_jaffe_tiff(folder_name, 
                lbl_dict = {'HA' : 0, 'SA' : 1, 'SU' : 2, 'AN' : 3, 'DI' : 4, 'FE' : 5, 'NE' : 6}):
    pass