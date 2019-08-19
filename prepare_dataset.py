import pandas as pd
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from face_detector import init_model_dnn, find_faces_dnn
from load_pics import load_img


class readData:

    def __init__(self):
        self._jaffe_data = None
        self._kanade_data = None
        self.conf_threshold=0.97


    def load_kanade(self, label_dir, img_dir, file_format='png'):
        """ Loads kanade dataset

            in: 
                * img_dir - path to database (images); 
                * label_dir - path to image labels;   
                * file_format - image format;                 

            Emotion labeling: 0 : neutral, 1 : anger, 2 : contempt, 
                3 : disgust, 4 : fear, 5 : happy, 6 : sadness, 7 : surprise 
        """
        labels = []
        data_path = []
        bbox = []      
        bbox_norm = []

        net = init_model_dnn()
        # Firstly we find out if image is labeled and if yes we load it to the database
        for root, dirs, files in os.walk(label_dir):
            for file in files:
                if '.txt' in file:
                    labeled_pics_path = os.path.join(root, file)
                    #? images will be in a similare directory, 
                    #? except file format and root directory will be different 
                    pic_file = file.replace('_emotion.txt', "." + file_format)
                    pic_root = root.replace(label_dir, img_dir)
                    pics_path = os.path.join(pic_root, pic_file)
                    im = load_img(pics_path, False, None)
                
                    # we fiind face on the image and save it to the database
                    detections = find_faces_dnn(im, net=net)
                    
                    confidence_max_pos = np.argmax(detections[0, 0, :, 2])

                    if detections[0, 0, confidence_max_pos, 2] > self.conf_threshold:
                        (h, w) = im.shape[:2]
                        box_norm =  detections[0, 0, 0, 3:7]
                        box = box_norm * np.array([w, h, w, h])
                        box = box.astype("int")
                        bbox.append(box)
                        bbox_norm.append(box_norm)
                        # (startX, startY, endX, endY)

                        data_path.append(pics_path)
                        with open(labeled_pics_path, 'r') as lbl:
                            labels.append(float(lbl.read()[:-1]) - 1.0)

        self._kanade_data = {'file':data_path, 'label': labels, 
                                'x0' : bbox[:, 0], 'y0' : bbox[:, 1], 
                                'x1' : bbox[:, 2], 'y1' : bbox[:, 3], 
                                'x_norm0' : box_norm[:, 0], 'y_norm0' : box_norm[:, 1], 
                                'x_norm1' : box_norm[:, 2], 'y_norm1' : box_norm[:, 3]                                
        }
    

    def load_jaffe(self, img_dir, file_format='tiff', 
                lbl_dict = {'NE' : 0, 'AN' : 1, 'DI' : 3, 'FE' : 4, 'HA' : 5, 'SA' : 6, 'SU' : 7 }):
        assert isinstance(img_dir, str) and isinstance(file_format, str) 
        f_list = glob.glob(img_dir + "/*." + file_format)
        labels = []
        data_path = []
        bbox = []      
        bbox_norm = []
        net = init_model_dnn()
        strt_ind = f_list[0].find('jaffe') + 9
        for file in f_list:
            im = load_img(file, False, None)
            detections = find_faces_dnn(im, net=net)
            confidence_max_pos = np.argmax(detections[0, 0, :, 2])
            if detections[0, 0, confidence_max_pos, 2] > self.conf_threshold:
                (h, w) = im.shape[:2]
                box_norm =  detections[0, 0, 0, 3:7]
                box = box_norm * np.array([w, h, w, h])
                box = box.astype("int")
                bbox.append(box)
                bbox_norm.append(box_norm)
                # (startX, startY, endX, endY)

                data_path.append(file)
                labels.append(lbl_dict[file[strt_ind : strt_ind + 2]])
        self._jaffe_data = {'file':data_path, 'label': labels, 
                            'x0' : bbox[:, 0], 'y0' : bbox[:, 1], 
                            'x1' : bbox[:, 2], 'y1' : bbox[:, 3], 
                            'x_norm0' : box_norm[:, 0], 'y_norm0' : box_norm[:, 1], 
                            'x_norm1' : box_norm[:, 2], 'y_norm1' : box_norm[:, 3]}


    def save_csv(self, filename):
        df = pd.DataFrame(data={
            'file':self._kanade_data['file'] + self._jaffe_data['file'], 
            'label': self._kanade_data['label'] + self._jaffe_data['label'], 

            'x0': self._kanade_data['x0'] + self._jaffe_data['x0'], 
            'y0': self._kanade_data['y0'] + self._jaffe_data['y0'], 
            'x1': self._kanade_data['x1'] + self._jaffe_data['x1'], 
            'y1': self._kanade_data['y1'] + self._jaffe_data['y1'], 

            'x_norm0': self._kanade_data['x_norm0'] + self._jaffe_data['x_norm0'], 
            'y_norm0': self._kanade_data['y_norm0'] + self._jaffe_data['y_norm0'], 
            'x_norm1': self._kanade_data['x_norm1'] + self._jaffe_data['x_norm1'], 
            'y_norm1': self._kanade_data['y_norm1'] + self._jaffe_data['y_norm1'], 
        })

        df.sample(6)
        df.to_csv(filename)    

