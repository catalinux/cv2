import cv2 as cv
import argparse
import numpy as np

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Threshold Demo'


def Threshold_Demo(val):
    # 0: Binary
    # 1: Binary Inverted
    # 2: Threshold Truncated
    # 3: Threshold to Zero
    # 4: Threshold to Zero Inverted
    threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv.getTrackbarPos(trackbar_value, window_name)
    dst = cv.adaptiveThreshold(src_gray, threshold_value, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    print(threshold_type, threshold_value)
    cv.imshow(window_name, np.hstack([dst, src_gray]))


parser = argparse.ArgumentParser(description='Code for Basic Thresholding Operations tutorial.')
parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
args = parser.parse_args()
src = cv.imread('table.jpg')


def show_thresh_adaptive(gray_enhance):
    while (1):
        cv.imshow('image', gray_enhance)

        # get current positions of four trackbars
        A = max(3, 1 + 2 * cv.getTrackbarPos('B1', 'track'))
        B = cv.getTrackbarPos('M1', 'track')
        C = max(3, 1 + 2 * cv.getTrackbarPos('B', 'track'))
        D = cv.getTrackbarPos('M', 'track')
        adap = cv.getTrackbarPos('M/G', 'track')
        blur_size = 2 * cv.getTrackbarPos('blur_size', 'track') + 1

        if adap == 0:
            adap = cv.ADAPTIVE_THRESH_MEAN_C
        else:
            adap = cv.ADAPTIVE_THRESH_GAUSSIAN_C
        # blurred = cv.GaussianBlur(gray_enhance, (blur_size, blur_size), 0)
        thresh = cv.adaptiveThreshold(gray_enhance, 255, adap, cv.THRESH_BINARY, A, B)
        # thresh2 = cv.adaptiveThreshold(blurred, 255, adap, cv.THRESH_BINARY, C, D)

        cv.imshow('thresh', thresh)
        # cv.imshow('thresh_with_blur', thresh2)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break


gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
show_thresh_adaptive(gray)
