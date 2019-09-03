import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import os

def add_emoji_to_image(img, emoji, bounding_box): 
    result = img.copy()
    
    emoji_png = emoji + '.png'
    emoji_path = os.path.join('emoji', emoji_png) 
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    
    center = (bounding_box[0] + bounding_box[1]) // 2
    
    x0, y0 = bounding_box[0, 0], bounding_box[0, 1]
    x1, y1 = bounding_box[1, 0], bounding_box[1, 1]
    
    size = y1 - y0
    
    emoji = cv2.resize(emoji, (size, size))

    alpha_emoji = emoji[:, :, 3] / 255.0
    alpha_emoji = np.array(alpha_emoji)
    alpha_img = 1.0 - alpha_emoji

    for c in range(0, 3):
        result[y0:y1, y0:y1, c] = alpha_emoji * emoji[:, :, c] + alpha_img * img[y0:y1, y0:y1, c]

    return result
