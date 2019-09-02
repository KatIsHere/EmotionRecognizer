import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

# working with image

emoji = cv2.imread('Smile.png', cv2.IMREAD_UNCHANGED)
img = cv2.imread('Test.jpg', cv2.IMREAD_COLOR)
cv2.imwrite('Test.png', img)
img = cv2.imread('Test.png', cv2.IMREAD_UNCHANGED)

height, width = img.shape[:2]
print(width, height)
y_offset = int(height / 2)
x_offset = int(width/2)

def add_emoji_to_image(emoji, img, x_offset, y_offset): 
    result = img.copy()

    y1, y2 = y_offset - emoji.shape[0] // 2, y_offset + emoji.shape[0] // 2   
    x1, x2 = x_offset - emoji.shape[1] // 2, x_offset + emoji.shape[1] // 2

    alpha_emoji = emoji[:, :, 3] / 255.0
    alpha_img = 1.0 - alpha_emoji

    for c in range(0, 3):
        result[y1:y2, x1:x2, c] = (alpha_emoji * emoji[:, :, c] + alpha_img * img[y1:y2, x1:x2, c])

    # print(result)
    cv2.imwrite('test_img.png', result)
    return 0

add_emoji_to_image(emoji, img, x_offset, y_offset)
