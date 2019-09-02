import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

# working with image

emoji_folder_path = ''
happy_emoji = cv2.imread('happy.png', cv2.IMREAD_UNCHANGED)
angry_emoji = cv2.imread('angry.png', cv2.IMREAD_UNCHANGED)
sad_emoji = cv2.imread('sad.png', cv2.IMREAD_UNCHANGED)
fear_emoji = cv2.imread('fear.png', cv2.IMREAD_UNCHANGED)
disgust_emoji = cv2.imread('disgust-2.png', cv2.IMREAD_UNCHANGED)
surprized_emoji = cv2.imread('surprized.png', cv2.IMREAD_UNCHANGED)


# we need png format of image to place emoji on it
# ------------------------------------------------------
# img = cv2.imread('Test.jpg', cv2.IMREAD_COLOR)
# cv2.imwrite('Test.png', img)
# img = cv2.imread('Test.png', cv2.IMREAD_UNCHANGED)
# -------------------------------------------------------



def add_emoji_to_image(emoji, img, bounding_box): 
    
    center_height, center_width = (bounding_box[0] + bounding_box[1])/2

    result = img.copy()

    y1, y2 = center_heigh - emoji.shape[0] // 2, center_heigh + emoji.shape[0] // 2   
    x1, x2 = center_width - emoji.shape[1] // 2, center_width + emoji.shape[1] // 2

    alpha_emoji = emoji[:, :, 3] / 255.0
    alpha_img = 1.0 - alpha_emoji

    for c in range(0, 3):
        result[y1:y2, x1:x2, c] = (alpha_emoji * emoji[:, :, c] + alpha_img * img[y1:y2, x1:x2, c])

    return result
