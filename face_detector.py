import cv2
import numpy as np

# TODO: this is a pretty bag crunch, need to find a workaround
path_to_haar = "C:\\ProgramData\\Miniconda3\\envs\\tf-gpu\\Lib\\site-packages\\cv2\\data\\"

def load_haar_cascade(face = True):
    global path_to_haar
    if face:
        cascades = cv2.CascadeClassifier(path_to_haar + 'haarcascade_frontalface_default.xml')
    else:
        cascades = cv2.CascadeClassifier( path_to_haar + 'haarcascade_eye.xml')
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
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    return net

def cut_out_faces_dnn(im, net=None, new_size=None, conf_threshold=0.7):
    if net is None:
        net = init_model_dnn()
    faces_found = []
    (h, w) = im.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
    
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > conf_threshold:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = im[startY:endY, startX:endX]
            if new_size is not None:
                assert len(new_size) == 2
                face = cv2.resize(face, new_size, interpolation = cv2.INTER_AREA)
            faces_found.append(face)
    return faces_found
    

def faces_from_database_dnn(x_data, new_size=None):
    net = init_model_dnn()
    new_data = []
    for img in x_data:
        found_face = np.array(cut_out_faces_dnn(np.array(img), net=net, new_size=new_size))
        new_data.append(found_face)
    return new_data

    



def cut_out_faces_HOG(img, new_size=None):
    pass

# TODO: write a solution using HOG detector
def faces_from_database_HOG(x_data, new_size=None):
    pass

