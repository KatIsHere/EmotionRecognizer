import pandas as pd
import os
import numpy as np
import random
import cv2
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import dlib
from pathlib import Path
import sys, os
ROOT_DIR = Path(__file__).parents[1]
sys.path.append(os.path.abspath(ROOT_DIR))
from face_detector.face_detector import init_model_dnn, find_faces_dnn
from utils.utils import convert_landmarks, load_img
 


class DataPreprocessor:
    #   0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

    def __init__(self):
        self._jaffe_data = None
        self._kanade_data = None
        self._facesdb_data = None
        self.conf_threshold=0.97

    # for each labeled foder loads first and two last images (the position of the face usually stays the same)
    def load_kanade(self, label_dir, img_dir, file_format='png', 
                    netral_percentage=0.2, 
                    label_map = {0 : 'neutral', 1: 'anger', 3 : 'disgust', 4 : 'fear', 5: 'happy', 6 : 'sadness', 7 : 'surprise', 2 : None}):
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
                                if label_map[labl] is not None:
                                    labels.append(label_map[labl])

                        if random.random() >= netral_percentage:    
                            # saves first neutral frame
                            bbox.append(box)
                            bbox_norm.append(box_norm)
                            data_path.append(pics_neutral_path)
                            labels.append(label_map[0])

        bbox = np.array(bbox)
        bbox_norm = np.array(bbox_norm)
        self._kanade_data = {'file': data_path, 'label': labels, 
                            'x0' : bbox[:, 0], 'y0' : bbox[:, 1], 
                            'x1' : bbox[:, 2], 'y1' : bbox[:, 3], 
                            'x_norm0' : bbox_norm[:, 0], 'y_norm0' : bbox_norm[:, 1], 
                            'x_norm1' : bbox_norm[:, 2], 'y_norm1' : bbox_norm[:, 3]                                
        }

    def load_facesdb(self, img_dir, file_format='tif', 
                    label_map = {0 : 'neutral', 1: 'happy', 2 : 'sadness', 3 : 'surprise', 4: 'anger', 5 : 'disgust', 6 : 'fear'}):
        # label mapping     no contempt     with contempt   |       New data
        # 0 - neutral    ->     0        ->      0          |           6
        # 1 - happy      ->     4        ->      5          |           3
        # 2 - sadness    ->     5        ->      6          |           4
        # 3 - surprise   ->     6        ->      7          |           5
        # 4 - anger      ->     1        ->      1          |           0
        # 5 - disgust    ->     2        ->      3          |           1
        # 6 - fear       ->     3        ->      4          |           2
        # 9 - kiss
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
                label_map = {'NE' : 'neutral', 'AN' : 'anger', 'DI' : 'disgust', 'FE' : 'fear', 'HA' : 'happy', 'SA' : 'sadness', 'SU' : 'surprise' }):
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


    def load_custom(self, img_dir, csv_file, csv_newfile,
                    label_map = {'anger':0, 'surprise':5, 'disgust':1, 'fear':2, 'neutral':6, 'happiness':3, 'sadness':4, 
                    'ANGER':0, 'SURPRISE':5, 'DISGUST':1, 'FEAR':2, 'NEUTRAL':6, 'HAPPINESS':3, 'SADNESS':4}):
        df = pd.read_csv(csv_file)
        bbox = []
        net = init_model_dnn()
        labels = []
        data_path = []
        for index, row in df.iterrows():
            im = load_img(img_dir + row['image'], False, None)
            detections = find_faces_dnn(im, net=net)
            confidence_max_pos = np.argmax(detections[0, 0, :, 2])

            if detections[0, 0, confidence_max_pos, 2] > self.conf_threshold:
                    (h, w) = im.shape[:2]
                    box_norm =  detections[0, 0, 0, 3:7]
                    if (box_norm <= [1.0, 1.0, 1.0, 1.0]).all() and label_map[row['emotion']] is not None:
                        box = box_norm * np.array([w, h, w, h])
                        bbox.append(box.astype("int"))
                        labels.append(label_map[row['emotion']])
                        data_path.append(img_dir + row['image'])

        bbox = np.array(bbox)
        labels = np.array(labels)
        df = pd.DataFrame(data={'file': data_path, 'label': labels, 
                            'x0' : bbox[:, 0], 'y0' : bbox[:, 1], 
                            'x1' : bbox[:, 2], 'y1' : bbox[:, 3]})
        assert isinstance(csv_newfile, str)
        df.to_csv(csv_newfile)

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


# 'contempt' is excluded from the dataset
def prepare_data_no_contempt():
    dt = DataPreprocessor()
    # dt.load_custom(os.path.join(ROOT_DIR,'data\\images\\'), os.path.join(ROOT_DIR,'data\\legend.csv'), 
    #               os.path.join(ROOT_DIR,'data\\legend_with_bboxes.csv'))
    dt.load_kanade(os.path.join(ROOT_DIR,"data\\kanade\\emotion\\"), 
                    os.path.join(ROOT_DIR,"data\\kanade\\cohn-kanade-images\\"))
    # #dt.save_csv(os.path.join(ROOT_DIR,'data\\dataset_kanade.csv')
    dt.load_facesdb(os.path.join(ROOT_DIR,'data\\facesdb\\'))
    dt.load_jaffe(os.path.join(ROOT_DIR,"data\\jaffe\\"))
    dt.save_csv(os.path.join(ROOT_DIR,'data\\dataset.csv'))
    # dt.clear()
    # dt.load_facesdb(os.path.join(ROOT_DIR,'data\\facesdb\\'))
    # dt.save_csv(os.path.join(ROOT_DIR,'data\\dataset_facesdb.csv'))
    # dt.clear()
    # dt.load_jaffe(os.path.join(ROOT_DIR,"data\\jaffe\\"))
    # dt.save_csv(os.path.join(ROOT_DIR,'data\\dataset_jaffe.csv'))


def localize_features(img, detector, predictor):
    rects = detector(img, 1)
    landmarks = []
    for k, rect in enumerate(rects):
        shape = predictor(img, rect)
        landmarks = convert_landmarks(rect, shape)
    if len(landmarks) != 0:
        return ' '.join(str(item) for innerlist in landmarks for item in innerlist)
    else:
        return None

def apply_augmentation(img, augment_code):
    h, w = img.shape[:2]
    if augment_code==0:
        return cv2.flip(img, 0)
    if augment_code==1:
        M = cv2.getRotationMatrix2D((w//2, h//2), int(random.random() * 100) % 35 , 1.0)
        return cv2.warpAffine(img, M, (w, h)) 


def prepare_dataset_with_dlib(csv_filename):  
    random.seed()  
    df = pd.read_csv(csv_filename)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(ROOT_DIR, 'face_detector\\shape_predictor_68_face_landmarks.dat'))
    features = []
    random_augmentation = []
    for index, row in df.iterrows():
        img = cv2.imread(row['file'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features.append(localize_features(img, detector, predictor))
        random_augmentation.append(localize_features(apply_augmentation(img, int(random.random() * 10) % 2), detector, predictor))
    df = df.assign(features=features)
    df = df.assign(random_augmentation=random_augmentation)
    df.to_csv(csv_filename)


def prepare_kadle_dataset_with_dlib(csv_filename, label_map):
    df = pd.read_csv(csv_filename)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(ROOT_DIR, 'face_detector\\shape_predictor_68_face_landmarks.dat'))
    x_data, y_data = [], []
    features = []
    for index, row in df.iterrows():
        img = np.fromstring(row['pixels'], sep=' ').astype('uint8')
        img = np.resize(img, (48, 48))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        rects = detector(img, 1)
        for k, rect in enumerate(rects):
                shape = predictor(img, rect)
                landmarks = convert_landmarks(rect, shape)
        x_data.append(landmarks)
        y_data.append(label_map[row['label']])
        features.append(' '.join(str(item) for innerlist in landmarks for item in innerlist))
    df = df.assign(features=features)
    df.to_csv(csv_filename)
    x_data, y_data = shuffle(x_data, y_data, random_state=random.seed())
    return  np.array(x_data),  np.array(y_data)

if __name__ == "__main__":
    random.seed()
    #plot_density_2(os.path.join(ROOT_DIR,'data\\dataset.csv'))
    #dt = DataPreprocessor()
    #dt.load_kanade(os.path.join(ROOT_DIR,"data\\kanade\\emotion\\"), 
    #               os.path.join(ROOT_DIR,"data\\kanade\\cohn-kanade-images\\"), 
    #               include_contempt=True)
    #dt.load_and_pross_fer2013(os.path.join(ROOT_DIR,'data\\fer2013.csv'), 
    #               os.path.join(ROOT_DIR,'data\\fer2013_bbox.csv'))

    print("reading data...")
    prepare_dataset_with_dlib(os.path.join(ROOT_DIR,'data\\dataset.csv'))
    #prepare_data_no_contempt()
    print("data saved")
