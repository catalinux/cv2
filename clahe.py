import cv2
import matplotlib.pyplot as plt
import numpy as np
from proiect.lib.util import imageshow, auto_canny

image_path = 'table.jpg'

bgr = cv2.imread(image_path)


def clahe(bgr, gridsize=11):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


imageshow(clahe(bgr))
