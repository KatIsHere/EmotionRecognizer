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
    
    y0, y1 = bounding_box[0, 1], bounding_box[1, 1]
    size = y1 - y0
    x0, x1 = max(center[0] - size // 2, 0), center[0] + (size + 1) // 2
  
    emoji = cv2.resize(emoji, (size, size))

    alpha_emoji = emoji[:, :, 3] / 255.0
    alpha_img = 1.0 - alpha_emoji

    for c in range(0, 3):
        result[y0:y1, x0:x1, c] = alpha_emoji * emoji[:, :, c] + alpha_img * img[y0:y1, x0:x1, c]

    return result


def add_all_emojis(img, list_emojis, list_bounding_boxes):
    labels_map = {0:'neutral', 1:'angry', 2:'disgust', 3:'fear', 4:'happy', 5:'sad', 6:'surprised'}
    n = list_emojis.shape[0]
    
    for i in range(n):
      emoji = labels_map[list_emojis[i]]
      if (emoji != 'neutral'):
        img = add_emoji_to_image(img, emoji, list_bounding_boxes[i])
        
    return img
