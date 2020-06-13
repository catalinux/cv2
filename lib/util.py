import cv2 as cv  # import
import imutils
import numpy as np
import math

SHOW_INTERMEDIATE_RESULTS = True


def imageshow(image, window_name='image'):
    if SHOW_INTERMEDIATE_RESULTS == True:
        cv.imshow(window_name, image)
        wait_for_key()
    pass


def wait_for_key():
    cv.waitKey(0)  # wait untill any keypress
    cv.destroyAllWindows()  # close the window
    cv.waitKey(1)
    cv.waitKey(1)
    cv.waitKey(1)
    cv.destroyAllWindows()

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    # return the edged image
    return edged
