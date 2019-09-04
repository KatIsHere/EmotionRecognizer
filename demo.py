import numpy as np
import cv2
import random
from keras.models import Sequential, model_from_json, Model 
import dlib
from place_emoji import add_markings
from utils.utils import convert_landmarks


def features_dlib(predictor_path='face_detector\\shape_predictor_68_face_landmarks.dat'):
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    #win = dlib.image_window()
    while True:
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #win.clear_overlay()
        #win.set_image(img)
        dets = detector(img)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
            #                                        shape.part(1)))
            # Draw the face landmarks on the screen.
            #win.add_overlay(shape)

        #win.add_overlay(dets)
        vec = np.empty([68, 2], dtype = int)
        for b in range(68):
            vec[b][0] = shape.part(b).x
            vec[b][1] = shape.part(b).y

        cv2.imshow('vid', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def run_facial_classifier():
    with open('saved_models\\dlib_facial.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights('saved_models\\dlib_facial.h5')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_detector\\shape_predictor_68_face_landmarks.dat')
    label_map = {0:'neutral', 1:'angry', 2:'disgust', 3:'fear', 4:'happy', 5:'sad', 6:'surprised'}

    cap = cv2.VideoCapture(0)
    vid_cod = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter("videos\\cam_video.mp4", vid_cod, 20.0, (640,480))
    cv2.namedWindow("demo",cv2.WINDOW_FULLSCREEN)
    while(True):
        ret, frame = cap.read()
        frame_clean = frame.copy()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = detector(img, 1)
        landmarks = []
        bboxes = []
        for k, rect in enumerate(rects):
            shape = predictor(img, rect)
            landmark = convert_landmarks(rect, shape)
            landmark = np.array(landmark).reshape((68, 2))
            landmarks.append(landmark)
            bboxes.append([(rect.left(), rect.top()), (rect.right(), rect.bottom())])

        landmarks = np.array(landmarks)
        bboxes = np.array(bboxes)
        if landmarks.shape[0] != 0:
            preds = model.predict(landmarks)
            preds = np.argmax(preds, axis=1)
            #preds = np.reshape(preds, (-1, 1))
            preds = np.array(preds)
            frame = add_markings(frame, preds, bboxes, place_emodji=True)
        frame = np.concatenate((frame, frame_clean))
        cv2.imshow('demo', frame)
        #output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    output.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    run_facial_classifier()
    #features_dlib()
    #detect_features()
    #classify_faces()