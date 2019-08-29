import numpy as np
import cv2
from keras_model import Emotion_Net
from facial_feature_extractor import Facial_Feature_Net, detect_and_find_features
from img_processor import detect_and_classify
import random

def detect_features():
    im_rows, im_cols, channels = 100, 100, 1
    
    Model = Facial_Feature_Net()
    Model.load_model("models\\facial_model.json")
    Model.load_weights("models\\facial_model_sm_2.h5")

    cap = cv2.VideoCapture(0)
    
    while(True):
        ret, frame = cap.read()

        frames = detect_and_find_features(frame, Model, new_size=(im_rows, im_cols))
        if frames is None:
            cv2.imshow('face : 0', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            i = 0
            for frame in frames: 
                cv2.imshow('face : ' + str(i), frame)
                i += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def classify_faces():
    cap = cv2.VideoCapture(0)
    model = Emotion_Net()
    model.load_model("models\\ins_resnet_v2_newdata_model.json")
    model.load_weights("models\\ins_resnet_v2_newdata_model.h5")

    while(True):
        ret, frame = cap.read()

        frame = detect_and_classify(frame, model, new_size=(96, 96), channels=3)

        cv2.imshow('vid', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    detect_features()