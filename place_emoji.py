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
    height_emoji = y1 - y0
    
    width_image = img.shape[1]  
    x0, x1 = max(center[0] - height_emoji // 2, 0), min(center[0] + (height_emoji + 1) // 2, width_image)     # rectangle boundings check 
    width_emoji = x1 - x0

    emoji = cv2.resize(emoji, (height_emoji, height_emoji))
    if(width_emoji < height_emoji):
      if (x0 == 0):
        emoji = emoji[:, height_emoji - width_emoji :, :]
      else:
        emoji = emoji[:, : width_emoji, :]

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
            img = add_bounding_box_with_label(img, emotion, list_bounding_boxes[i])        
    return img
