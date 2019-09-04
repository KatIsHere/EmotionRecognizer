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
    height = y1 - y0
    width = bounding_box[1, 0] - bounding_box[0, 0]  
    x0, x1 = max(center[0] - height // 2, 0), min(center[0] + (height + 1) // 2, width)     # rectangle boundings check
  
    emoji = cv2.resize(emoji, (height, height))

    alpha_emoji = emoji[:, :, 3] / 255.0
    alpha_img = 1.0 - alpha_emoji

    for c in range(0, 3):
        result[y0:y1, x0:x1, c] = alpha_emoji * emoji[:, :, c] + alpha_img * img[y0:y1, x0:x1, c]

    return result


def add_bounding_box_with_label(img, emoji, bounding_box):
    result = cv2.rectangle(img, (bounding_box[0][0], bounding_box[0][1]), 
                        (bounding_box[1][0], bounding_box[1][1]), (0, 0, 255), 2)
    result = cv2.putText(result, emoji, (bounding_box[0][0], bounding_box[0][1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10, 255, 0), thickness=2)
    return result


def add_markings(img, list_emojis, list_bounding_boxes, place_emodji=True, 
                labels_map = {0:'neutral', 1:'angry', 2:'disgust', 
                        3:'fear', 4:'happy', 5:'sad', 6:'surprised'}
                ):
    n = list_emojis.shape[0]

    for i in range(n):
        emotion = labels_map[list_emojis[i]]
        if place_emodji and emotion != 'neutral':
                img = add_emoji_to_image(img, emotion, list_bounding_boxes[i])
        else:   # for testing boundig box and classifier
            img = add_markings(img, emotion, list_bounding_boxes[i])        
    return img