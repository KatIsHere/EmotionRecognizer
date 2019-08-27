import numpy as np
import cv2
from keras_model import Emotion_Net
from img_processor import detect_and_classify

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    model = Emotion_Net()
    model.load_model("models\\combined_model.json")
    model.load_weights("models\\combined_model.h5")

    while(True):
        ret, frame = cap.read()

        frame = detect_and_classify(frame, model, new_size=(200, 200), channels=1)

        cv2.imshow('vid', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()