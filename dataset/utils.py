import cv2
import random
import numpy as np


def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
         cumulative_probability += item_probability
         if x < cumulative_probability:
               break
    return item


def augment_brightness_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 0] * (0.75 + np.random.random() * 0.5)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.75 + np.random.random() * 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.75 + np.random.random() * 0.5)
    image1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image1

def augment_horizontal_flip(image, rois):
    image1 = image.copy()
    image1 = image1[:, ::-1]
    rois1 = rois.copy()
    rois1[:, 0] = 1.0 - rois[:, 2]
    rois1[:, 2] = 1.0 - rois[:, 0]
    return image1, rois1

def augment_jitter_rois(rois, jitter_max=0.1):
    rois1 = rois.copy()
    N = len(rois1)
    dx1 = (rois1[:, 2] - rois1[:, 0]) * np.random.uniform(-jitter_max, jitter_max, N)
    dy1 = (rois1[:, 3] - rois1[:, 1]) * np.random.uniform(-jitter_max, jitter_max, N)
    dx2 = (rois1[:, 2] - rois1[:, 0]) * np.random.uniform(-jitter_max, jitter_max, N)
    dy2 = (rois1[:, 3] - rois1[:, 1]) * np.random.uniform(-jitter_max, jitter_max, N)
    rois1 = rois1 + np.stack((dx1, dy1, dx2, dy2), axis=1)
    return rois1
