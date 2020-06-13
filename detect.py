import numpy as np
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt
from proiect.lib.util import imageshow, auto_canny

templates = []
base_folder_matching = '../Lab6/template_matching'
images_names = glob.glob(os.path.join(base_folder_matching, "*.jpg"))
for image_name in images_names:
    template = cv.imread(image_name)
    templates.append(template)
    cv.imshow("template", template)
    cv.waitKey(2000)
    cv.destroyAllWindows()

color_dict = {0: "black",
              1: "blue",
              2: "brown",
              3: "green",
              4: "pink",
              5: "red",
              6: "white",
              7: "yellow"}


frame = cv.imread('table2.jpg')
first_frame = frame.copy()
idx = -1
for template in templates:
    idx = idx + 1
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    res = cv.matchTemplate(frame_gray, template_gray, cv.TM_CCOEFF_NORMED)

    imageshow(res,"res")
    threshold = 0.75
    loc = np.where(res >= threshold)
    frame_draw = first_frame.copy()
    for pt in zip(*loc[::-1]):
        cv.rectangle(frame_draw, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    print(color_dict[idx])
    imageshow(frame_draw,"Template_matching " + color_dict[idx])
