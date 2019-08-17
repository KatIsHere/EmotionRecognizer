import cv2
import numpy as np


def load_haar_cascade(face = True):
    if face:
        cascades = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    else:
        cascades = cv2.CascadeClassifier('haarcascade_eye.xml')
    return cascades


def find_faces_on_img(img, cascades = None, param1 = 1.5, param2 = 5):
    if cascades is None:
        cascades = load_haar_cascade()
    assert len(img.shape) == 2 or img.shape[-1] == 1
    faces = cascades.detectMultiScale(img, param1, param2)
    return faces    # [(x, y, w, h) rects]



