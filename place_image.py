import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

# working with image

emoji = cv2.imread('Smile.png', cv2.IMREAD_UNCHANGED)
img = cv2.imread('Test.jpg', cv2.IMREAD_COLOR)
cv2.imwrite('Test.png', img)
img = cv2.imread('Test.png', cv2.IMREAD_UNCHANGED)

