import cv2
import numpy as np

def load_haar_cascade(face = True):
    if face:
        cascades = cv2.CascadeClassifier('face_detection\\haarcascade_frontalface_default.xml')
    else:
        cascades = cv2.CascadeClassifier('face_detection\\haarcascade_eye.xml')
    return cascades


def find_faces_on_img(img, cascades = None, param1 = 1.5, param2 = 5):
    if cascades is None:
        cascades = load_haar_cascade()
    assert len(img.shape) == 2 or img.shape[-1] == 1
    faces = cascades.detectMultiScale(img, param1, param2)
    return faces    # [(x, y, w, h) rects]

def cut_out_faces_haar(img, cascades=None, new_size=None):
    faces = find_faces_on_img(img, cascades)
    faces_found = []
    for (x, y, w, h) in faces:
        if len(img.shape) == 3:
            face = img[y:y+h, x:x+w, :]
        else:
            face = img[y:y+h, x:x+w]
            
        if new_size is not None:
            assert len(new_size) == 2
            face = cv2.resize(face, new_size, interpolation = cv2.INTER_AREA)
        faces_found.append(face)
    return faces_found

def faces_from_database_haar(x_data, new_size=None):
    new_data = []
    cascades= load_haar_cascade()
    for img in x_data:
        found_face = np.array(cut_out_faces_haar(np.array(img), cascades=cascades, new_size=new_size))
        new_data.append(found_face)
    return new_data

def init_model_dnn():
    modelFile = "face_detection\\opencv_face_detector_uint8.pb"
    configFile = "face_detection\\opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    return net

def cut_out_faces_dnn(im, net=None, new_size=None, to_greyscale=False, conf_threshold=0.98):
    if net is None:
        net = init_model_dnn()
    faces_found = []
    (h, w) = im.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = im[startY:endY, startX:endX]

            if to_greyscale:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            if new_size is not None:
                assert len(new_size) == 2
                face = cv2.resize(face, new_size, interpolation = cv2.INTER_AREA)
            faces_found.append(face)

    return np.array(faces_found)
    

def find_faces_dnn(im, net=None):
    if net is None:
        net = init_model_dnn()
    blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
            
    return detections

def faces_from_database_dnn(x_data, new_size=None, to_greyscale=False):
    net = init_model_dnn()
    new_data = []
    for img in x_data:
        found_face = np.array(cut_out_faces_dnn(np.array(img), net=net, new_size=new_size, to_greyscale=to_greyscale))
        #new_data = np.vstack([new_data, found_face])
        new_data.append(found_face)
    return new_data

    



# TODO: write a solution using HOG detector
def cut_out_faces_HOG(img, new_size=None):
    pass

def faces_from_database_HOG(x_data, new_size=None):
    pass

