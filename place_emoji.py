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
    x0, x1 = max(center[0] - size // 2, 0), center[0] + size // 2
  
    emoji = cv2.resize(emoji, (size, size))

    alpha_emoji = emoji[:, :, 3] / 255.0
    alpha_emoji = np.array(alpha_emoji)
    alpha_img = 1.0 - alpha_emoji

    for c in range(0, 3):
        result[y0:y1, x0:x1, c] = alpha_emoji * emoji[:, :, c] + alpha_img * img[y0:y1, x0:x1, c]

    return result

def add_all_emojis(img, list_emojis, list_bounding_boxes):
  label_map = {}
  n = len(list_emojis)
  
  for i in range(n):
    emoji = label_map(list_emojis[i])
    bounding_box = list_bounding_boxes[i]
    if(emoji != 'neutral'):
      img = add_emoji_to_image(img, emoji, bounding_box)
      
  return img
