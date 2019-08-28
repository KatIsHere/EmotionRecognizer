import numpy as np
import cv2
from keras_model import Emotion_Net
from img_processor import detect_and_classify
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('data\\fer2013.csv')
    df = df.head(1000)
    use_frame = []
    label_map = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
    for index, row in df.iterrows():
        im = np.array([int(x) for x in row['pixels'].split(' ')]).astype('uint8')
        im = np.reshape(im, (48, 48))
        im = cv2.resize(im, (48*5,48*5), cv2.INTER_AREA)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        im = cv2.putText(im, label_map[row['emotion']], (5, 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('vid', im)
        k = cv2.waitKey(0)
        if k == ord('1'):   
            use_frame.append(1)
        elif k == ord(2):  
            use_frame.append(0)
        else:
            use_frame.append(None)
            
    df = df.assign(use_img=use_frame)
    df.to_csv('data\\fer2013_filtered.csv')