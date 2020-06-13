import numpy as np
import cv2 as cv
import time

def clahe(bgr, gridsize=11):
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    lab_planes = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv.merge(lab_planes)
    bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    return bgr
    return bgr

img = cv.imread('./data/training_data/Task1/49.jpg', cv.IMREAD_COLOR)
# img = cv.imread('./table.jpg', cv.IMREAD_COLOR)
# img = cv.medianBlur(img, 5)
# img = clahe(img)
# Convert BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# 'LOWER': np.array([0, 136, 150]),
# 'UPPER': np.array([10, 225, 255])
lh = 0
ls = 0
lv = 240
uh = 255
us = 137
uv = 255
lower_hsv = np.array([lh, ls, lv])
upper_hsv = np.array([uh, us, uv])
mask = cv.inRange(hsv, lower_hsv, upper_hsv)
window_name = "HSV Calibrator"
cv.namedWindow(window_name)

def nothing(x):
    print("Trackbar value: " + str(x))
    pass

# create trackbars for Upper HSV
cv.createTrackbar('UpperH',window_name,0,255,nothing)
cv.setTrackbarPos('UpperH',window_name, uh)

cv.createTrackbar('UpperS',window_name,0,255,nothing)
cv.setTrackbarPos('UpperS',window_name, us)

cv.createTrackbar('UpperV',window_name,0,255,nothing)
cv.setTrackbarPos('UpperV',window_name, uv)

# create trackbars for Lower HSV
cv.createTrackbar('LowerH',window_name,0,255,nothing)
cv.setTrackbarPos('LowerH',window_name, lh)

cv.createTrackbar('LowerS',window_name,0,255,nothing)
cv.setTrackbarPos('LowerS',window_name, ls)

cv.createTrackbar('LowerV',window_name,0,255,nothing)
cv.setTrackbarPos('LowerV',window_name, lv)

font = cv.FONT_HERSHEY_SIMPLEX


while(1):

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    # cv.putText(mask,'Lower HSV: [' + str(lh) +',' + str(ls) + ',' + str(lv) + ']', (10,30), font, 0.5, (200,255,155), 1, cv.LINE_AA)
    # cv.putText(mask,'Upper HSV: [' + str(uh) +',' + str(us) + ',' + str(uv) + ']', (10,60), font, 0.5, (200,255,155), 1, cv.LINE_AA)

    im = cv.bitwise_and(img, img, mask=mask)
    # im = hsv[mask]
    cv.putText(mask, 'Lower HSV: [' + str(lh) + ',' + str(ls) + ',' + str(lv) + ']', (10, 30), font, 0.5,
               (200, 255, 155), 1, cv.LINE_AA)
    cv.putText(mask, 'Upper HSV: [' + str(uh) + ',' + str(us) + ',' + str(uv) + ']', (10, 60), font, 0.5,
               (200, 255, 155), 1, cv.LINE_AA)

    cv.imshow(window_name,im)
    # cv.imshow(window_name,mask)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of Upper HSV trackbars
    uh = cv.getTrackbarPos('UpperH',window_name)
    us = cv.getTrackbarPos('UpperS',window_name)
    uv = cv.getTrackbarPos('UpperV',window_name)
    upper_blue = np.array([uh,us,uv])
    # get current positions of Lower HSCV trackbars
    lh = cv.getTrackbarPos('LowerH',window_name)
    ls = cv.getTrackbarPos('LowerS',window_name)
    lv = cv.getTrackbarPos('LowerV',window_name)
    upper_hsv = np.array([uh,us,uv])
    lower_hsv = np.array([lh,ls,lv])

    time.sleep(.1)

cv.destroyAllWindows()