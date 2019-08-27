import pandas as pd
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from face_detector import init_model_dnn, find_faces_dnn
from load_pics import load_img
import random
import cv2

class DataPreprocessor:

    def __init__(self):
        self._jaffe_data = None
        self._kanade_data = None
        self._facesdb_data = None
        self.conf_threshold=0.97

    # for each labeled foder loads first and two last images (the position of the face usually stays the same)
    def load_kanade(self, label_dir, img_dir, file_format='png', netral_percentage=0.5, include_contempt=False):
        """ Loads kanade dataset

            in: 
                * img_dir - path to database (images); 
                * label_dir - path to image labels;   
                * file_format - image format;                 

            Emotion labeling: 0 : neutral, 1 : anger, 2 : contempt, 
                3 : disgust, 4 : fear, 5 : happy, 6 : sadness, 7 : surprise 
        """
        assert isinstance(label_dir, str) and isinstance(img_dir, str) and isinstance(file_format, str) 
        labels, data_path = [], []
        bbox, bbox_norm = [], []
        net = init_model_dnn()

        # Firstly we find out if image is labeled and if yes we load it to the database
        for root, dirs, files in os.walk(label_dir):
            for file in files:
                if '.txt' in file:
                    labeled_pics_path = os.path.join(root, file)
                    #? images will be in a similare directory, 
                    #? except file format and root directory will be different 
                    pic_file = file.replace('_emotion.txt', '')
                    pic_root = root.replace(label_dir, img_dir)
                    pics_path = os.path.join(pic_root, pic_file + "." + file_format)
                    pics_neutral_path = os.path.join(pic_root, pic_file[:-2] + "01." + file_format)    
                    im = load_img(pics_path, False, None)
                
                    # we fiind face on the image and save it to the database
                    detections = find_faces_dnn(im, net=net)
                    confidence_max_pos = np.argmax(detections[0, 0, :, 2])

                    if detections[0, 0, confidence_max_pos, 2] > self.conf_threshold:
                        (h, w) = im.shape[:2]
                        box_norm =  detections[0, 0, 0, 3:7]
                        box = box_norm * np.array([w, h, w, h])
                        box = box.astype("int") 

                        with open(labeled_pics_path, 'r') as lbl:
                            labl = float(lbl.read()[:-1])
                            if labl != 2.0:
                                bbox.append(box)
                                bbox_norm.append(box_norm)
                                data_path.append(pics_path)
                                if labl > 2.0 and not include_contempt:
                                    labels.append(labl - 1.)
                                else:
                                    labels.append(labl)

                        if random.random() >= netral_percentage:    
                            # saves first neutral frame
                            bbox.append(box)
                            bbox_norm.append(box_norm)
                            data_path.append(pics_neutral_path)
                            labels.append(0.0)

        bbox = np.array(bbox)
        bbox_norm = np.array(bbox_norm)
        self._kanade_data = {'file': data_path, 'label': labels, 
                            'x0' : bbox[:, 0], 'y0' : bbox[:, 1], 
                            'x1' : bbox[:, 2], 'y1' : bbox[:, 3], 
                            'x_norm0' : bbox_norm[:, 0], 'y_norm0' : bbox_norm[:, 1], 
                            'x_norm1' : bbox_norm[:, 2], 'y_norm1' : bbox_norm[:, 3]                                
        }

    def load_facesdb(self, img_dir, file_format='tif', 
                    label_map = {0 : 0, 1: 4, 2 : 5, 3 : 6, 4: 1, 5 : 2, 6 : 3}):
        # label mapping     no contempt     with contempt
        # 0 - Neutral    ->     0        ->      0
        # 1 - Happy      ->     4        ->      5
        # 2 - Sadness    ->     5        ->      6
        # 3 - Surprise   ->     6        ->      7
        # 4 - Anger      ->     1        ->      1
        # 5 - Disgust    ->     2        ->      3
        # 6 - Fear       ->     3        ->      4
        assert isinstance(img_dir, str) and isinstance(file_format, str) 
        labels, data_path = [], []
        bbox, bbox_norm = [], []
        net = init_model_dnn()

        for root, dirs, files in os.walk(img_dir):
            for file in files:
                strt_ind = file.find('_img.' + file_format) - 2
                label = int(file[strt_ind : strt_ind + 2])
                if label <= 6:
                    file_path = os.path.join(root, file)
                    im = load_img(file_path, False, None)
                    detections = find_faces_dnn(im, net=net)
                    confidence_max_pos = np.argmax(detections[0, 0, :, 2])
                    if detections[0, 0, confidence_max_pos, 2] > self.conf_threshold:
                        (h, w) = im.shape[:2]
                        box_norm =  detections[0, 0, 0, 3:7]
                        box = box_norm * np.array([w, h, w, h])
                        box = box.astype("int")

                        bbox.append(box)
                        bbox_norm.append(box_norm)
                        data_path.append(file_path)
                        labels.append(label_map[label])
        bbox = np.array(bbox)
        bbox_norm = np.array(bbox_norm)
        self._facesdb_data = {'file': data_path, 'label': labels, 
                            'x0' : bbox[:, 0], 'y0' : bbox[:, 1], 
                            'x1' : bbox[:, 2], 'y1' : bbox[:, 3], 
                            'x_norm0' : bbox_norm[:, 0], 'y_norm0' : bbox_norm[:, 1], 
                            'x_norm1' : bbox_norm[:, 2], 'y_norm1' : bbox_norm[:, 3]}
    
    def load_jaffe(self, img_dir, file_format='tiff', 
                label_map = {'NE' : 0, 'AN' : 1, 'DI' : 2, 'FE' : 3, 'HA' : 4, 'SA' : 5, 'SU' : 6 }):
        assert isinstance(img_dir, str) and isinstance(file_format, str) 
        f_list = glob.glob(img_dir + "/*." + file_format)
        labels, data_path = [], []
        bbox, bbox_norm = [], []
        net = init_model_dnn()

        strt_ind = f_list[0].find('jaffe') + 9  # this one is bad
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
                labels.append(label_map[file[strt_ind : strt_ind + 2]])
        bbox = np.array(bbox)
        bbox_norm = np.array(bbox_norm)
        self._jaffe_data = {'file': data_path, 'label': labels, 
                            'x0' : bbox[:, 0], 'y0' : bbox[:, 1], 
                            'x1' : bbox[:, 2], 'y1' : bbox[:, 3], 
                            'x_norm0' : bbox_norm[:, 0], 'y_norm0' : bbox_norm[:, 1], 
                            'x_norm1' : bbox_norm[:, 2], 'y_norm1' : bbox_norm[:, 3]}

    def load_and_pross_fer2013(self, csv_file, csv_newfile, im_size = (48, 48)):
        df = pd.read_csv(csv_file)
        bbox = []
        net = init_model_dnn()
        for index, row in df.iterrows():
            im = np.array([int(x) for x in row['pixels'].split(' ')]).astype('uint8')
            im = np.reshape(im, im_size)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            detections = find_faces_dnn(im, net=net)
            confidence_max_pos = np.argmax(detections[0, 0, :, 2])

            if detections[0, 0, confidence_max_pos, 2] > self.conf_threshold:
                    (h, w) = im_size
                    box_norm =  detections[0, 0, 0, 3:7]
                    box = box_norm * np.array([w, h, w, h])
                    bbox.append(box.astype("int"))

        bbox = np.array(bbox)
        df = df.assign(x0 = bbox[:, 0])
        df = df.assign(y0 = bbox[:, 1])
        df = df.assign(x1 = bbox[:, 2])
        df = df.assign(y1 = bbox[:, 3])
        assert isinstance(csv_newfile, str)
        df.to_csv(csv_newfile)

    def clear(self):
        del self._kanade_data
        self._kanade_data = None
        del self._jaffe_data
        self._jaffe_data = None
        del self._facesdb_data
        self._facesdb_data = None

    def save_csv(self, filename):
        """Saves all loaded data to .csv file"""
        assert isinstance(filename, str)
        data = None
        if self._kanade_data is not None:
            data = self._kanade_data
        
        if self._jaffe_data is not None and data is not None:
            data = {
                    'file': data['file'] + self._jaffe_data['file'], 
                    'label': data['label'] + self._jaffe_data['label'], 

                    'x0': np.concatenate((data['x0'], self._jaffe_data['x0'])), 
                    'y0': np.concatenate((data['y0'], self._jaffe_data['y0'])), 
                    'x1': np.concatenate((data['x1'], self._jaffe_data['x1'])), 
                    'y1': np.concatenate((data['y1'], self._jaffe_data['y1'])), 

                    'x_norm0': np.concatenate((data['x_norm0'], self._jaffe_data['x_norm0'])), 
                    'y_norm0': np.concatenate((data['y_norm0'], self._jaffe_data['y_norm0'])), 
                    'x_norm1': np.concatenate((data['x_norm1'], self._jaffe_data['x_norm1'])), 
                    'y_norm1': np.concatenate((data['y_norm1'], self._jaffe_data['y_norm1']))
                }
        elif self._jaffe_data is not None:
            data = self._jaffe_data

        if self._facesdb_data is not None and data is not None:
            data = {
                    'file': data['file'] + self._facesdb_data['file'], 
                    'label': data['label'] + self._facesdb_data['label'], 

                    'x0': np.concatenate((data['x0'], self._facesdb_data['x0'])), 
                    'y0': np.concatenate((data['y0'], self._facesdb_data['y0'])), 
                    'x1': np.concatenate((data['x1'], self._facesdb_data['x1'])), 
                    'y1': np.concatenate((data['y1'], self._facesdb_data['y1'])), 

                    'x_norm0': np.concatenate((data['x_norm0'], self._facesdb_data['x_norm0'])), 
                    'y_norm0': np.concatenate((data['y_norm0'], self._facesdb_data['y_norm0'])), 
                    'x_norm1': np.concatenate((data['x_norm1'], self._facesdb_data['x_norm1'])), 
                    'y_norm1': np.concatenate((data['y_norm1'], self._facesdb_data['y_norm1']))
                }
        elif self._facesdb_data is not None:
            data = self._facesdb_data

        if data is None:
            return
        df = pd.DataFrame(data=data)
        df.to_csv(filename)    


class PrepareMUCT:
    def __init__(self):
        self._datafr = None
        self.conf_threshold = 0.85

    def preproses(self, img_dir, csv_filename):
        self._datafr = pd.read_csv(csv_filename)
        bbox = []
        drop_rows = []
        net = init_model_dnn()
        self._datafr = self._datafr.head(3755)
        for index, row in self._datafr.iterrows():
            f_path = img_dir + row['name'] + '.jpg'
            im = load_img(f_path, False, None)
            if im is not None:
                row['name'] = f_path
                detections = find_faces_dnn(im, net=net)
                confidence_max_pos = np.argmax(detections[0, 0, :, 2])

                if detections[0, 0, confidence_max_pos, 2] > self.conf_threshold:
                    (h, w) = im.shape[:2]
                    box_norm =  detections[0, 0, 0, 3:7]
                    box = box_norm * np.array([w, h, w, h])
                    box = box.astype("int") 
                    box -= (3, 3, 0, 0)
                    box += (0, 0, 3, 3)
                    # im = cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    # im = cv2.putText(im, str(np.max(detections[0, 0, :, 2])), (box[0], box[1] - 5), 
                    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # cv2.imshow("test", im)
                    # cv2.waitKey(0)  
                    bbox.append(box)
                else:
                    drop_rows.append(index)
            else:
                drop_rows.append(index)
        self._datafr.drop(drop_rows)
        bbox = np.array(bbox)
        self._datafr = self._datafr.assign(bbox_x0 = bbox[:, 0])
        self._datafr = self._datafr.assign(bbox_y0 = bbox[:, 1])
        self._datafr = self._datafr.assign(bbox_x1 = bbox[:, 2])
        self._datafr = self._datafr.assign(bbox_y1 = bbox[:, 3])


    def save_csv(self, filename):
        """Saves all loaded data to .csv file"""
        assert isinstance(filename, str)
        self._datafr.to_csv(filename)


# 'contempt' is included in the dataset
def prepare_data_with_contempt():
    dt = DataPreprocessor()
    dt.load_kanade("data\\kanade\\emotion\\", "data\\kanade\\cohn-kanade-images\\", include_contempt=True)
    dt.save_csv('data\\dataset_kanade_8.csv')
    dt.load_facesdb('data\\facesdb\\', label_map = {0 : 0, 1: 5, 2 : 6, 3 : 7, 4: 1, 5 : 3, 6 : 4})
    dt.load_jaffe("data\\jaffe\\", label_map = {'NE' : 0, 'AN' : 1, 'DI' : 3, 'FE' : 4, 'HA' : 5, 'SA' : 6, 'SU' : 7 })
    dt.save_csv('data\\dataset_8.csv')
    dt.clear()
    dt.load_facesdb('data\\facesdb\\')
    dt.save_csv('data\\dataset_facesdb_8.csv')
    dt.clear()
    dt.load_jaffe("data\\jaffe\\", label_map = {'NE' : 0, 'AN' : 1, 'DI' : 3, 'FE' : 4, 'HA' : 5, 'SA' : 6, 'SU' : 7 })
    dt.save_csv('data\\dataset_jaffe_8.csv')


# 'contempt' is excluded from the dataset
def prepare_data_no_contempt():
    dt = DataPreprocessor()
    dt.load_kanade("data\\kanade\\emotion\\", "data\\kanade\\cohn-kanade-images\\", include_contempt=False)
    dt.save_csv('data\\dataset_kanade.csv')
    dt.load_facesdb('data\\facesdb\\')
    dt.load_jaffe("data\\jaffe\\")
    dt.save_csv('data\\dataset.csv')
    dt.clear()
    dt.load_facesdb('data\\facesdb\\')
    dt.save_csv('data\\dataset_facesdb.csv')
    dt.clear()
    dt.load_jaffe("data\\jaffe\\")
    dt.save_csv('data\\dataset_jaffe.csv')


if __name__ == "__main__":
    random.seed()
    dt = DataPreprocessor()
    #dt.load_kanade("data\\kanade\\emotion\\", "data\\kanade\\cohn-kanade-images\\", include_contempt=True)
    dt.load_and_pross_fer2013('data\\fer2013.csv', 'data\\fer2013_bbox.csv')

    # print("reading data...")
    # prepare_data_with_contempt()
    # print("data saved")    
    # print("reading data...")
    # prepare_data_no_contempt()
    # print("data saved")
