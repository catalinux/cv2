import math
import pickle

import numpy as np
import cv2 as cv
import imutils
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from proiect.lib.util import imageshow, auto_canny

# img = cv.imread('table.jpg')
img = cv.imread('data/training_data/Task1/11.jpg')
orig = img.copy()
pickle_in = open("dict.pickle", "rb")
example_dict = pickle.load(pickle_in)

print(example_dict)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray = cv.bilateralFilter(gray1, 75, 75, 25)
# imageshow(np.vstack([gray,gray1]))
thresh = cv.adaptiveThreshold(gray, 222, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 3)
imageshow(thresh)

diameterBall = 34
radiusBall = diameterBall / 2
circles = cv.HoughCircles(thresh, cv.HOUGH_GRADIENT, 1, diameterBall * 0.7,
                          minRadius=int(round(radiusBall * 0.85)),
                          maxRadius=int(round(radiusBall * 2)),
                          param1=200,
                          param2=10
                          )
print(circles)
board = gray.copy()
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    print("Size", len(circles))
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv.circle(board, (x, y), r, (0, 255, 0), 2)
        cv.rectangle(board, (x - 3, y - 3), (x + 3, y + 3), (0, 128, 255), -1)
imageshow(board)

cnts, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv.contourArea, reverse=True)
circles = []
for a in cnts:
    (x, y), radius = cv.minEnclosingCircle(a)
    if radius > 1 and radius < 40:
        print(radius)
        circles.append(a)
        cv.circle(img, (int(x), int(y)), int(radius), (0, 244, 0), 3)

imageshow(thresh)
# cv.drawContours(img,circles[:20],-1,(233,0,0),2)
imageshow(img)
M = cv.moments(example_dict)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
pass

new = []
factor = 0.9
for a in example_dict[:, 0]:
    new.append([[int((a[0] - cX) * factor + cX), int((a[1] - cY) * factor + cY)]])
#
# new = [new]
new = np.array(new)
# cv.drawContours(img,[example_dict],-1,(244,0,200),2)
# cv.drawContours(img,[new],-1,(244,0,200),2)
# imageshow(img)
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray, 50, 150, apertureSize=3)
# imageshow(edges)
# # cnts, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# #
# # # cv.drawContours(img,cnts,-1,(200,1,200),2)
# # cnts = sorted(cnts, key=cv.contourArea, reverse=True)
# # aprox_param = 0.015
# # rectangles = []
# # for c in cnts:
# #     # approximate the contour
# #     peri = cv.arcLength(c, True)
# #     approx = cv.approxPolyDP(c, aprox_param * peri, True)
# #     if len(approx) < 10:
# #         contourArea = cv.contourArea(c)
# #         # print("Area ", contourArea)
# #         rectangles.append(approx)
# #
# # hull_list = []
# # for i in range(len(cnts)):
# #     hull = cv.convexHull(cnts[i])
# #     hull_list.append(hull)
# #
# # # cv.drawContours(img, hull_list, 0, (0, 255, 0), 3)
# # cv.drawContours(img, rectangles, -1, (255, 0, 0), 3)
# #
# # imageshow(np.hstack([edges, gray]))
# #
# # c = example_dict
# # extLeft = tuple(c[c[:, :, 0].argmin()][0])
# # extRight = tuple(c[c[:, :, 0].argmax()][0])
# # extTop = tuple(c[c[:, :, 1].argmin()][0])
# # extBot = tuple(c[c[:, :, 1].argmax()][0])
# #
# # cv.drawContours(img, [c], -1, (0, 255, 255), 2)
# # cv.circle(img, extLeft, 8, (0, 0, 255), -1)
# # cv.circle(img, extRight, 8, (0, 255, 0), -1)
# # cv.circle(img, extTop, 8, (255, 0, 0), -1)
# # cv.circle(img, extBot, 8, (255, 255, 0), -1)
# #
# # imageshow(img)
#
#
# mask = img==[0,0,0]
# kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
# proc = cv.dilate(img, kernel)
#
# thresh=cv.dilate(edges,kernel,iterations=10)
#
# tmp = cv.HoughLinesP(thresh, 1, math.pi / 180,100,None,30,10)
#
# imageshow(thresh)
#
# #lines = cv.HoughLinesP(thresh, rho=1, theta=np.pi / 180, threshold=2, minLineLength=30, maxLineGap=10)
# lines = tmp
#
# print(tmp)
# print(lines)
# imageshow(thresh, "lines")
# for x1, y1, x2, y2 in lines[0]:
#     # cv.drawContours()
#     cv.line(img, (x1, y1), (x2, y2), (233, 255, 0), 4)
# imageshow(img, "Lines")
# pass

# peri = cv.arcLength(example_dict, True)
# approx = cv.approxPolyDP(example_dict, 0.03 * peri, True)
#
# cv.drawContours(img,[approx],-1,(213,11,11),3)
# imageshow(img)

board = img.copy()
edges = cv.Canny(thresh, 75, 150)
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 10, maxLineGap=50, minLineLength=200)

vert = []
slopes = []
intercept = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = 0
        if x2 != x1:
            slope = (float(y2 - y1) / (float(x2 - x1)))
        color = (0, 255, 0)
        if slope < 0.3 and slope > -0.3:
            color = (222, 0, 2)
        else:
            print(slope)

            slopes.append(slope)
            y2new = img.shape[0]
            x2new = int((y2new - y1) / slope + x1)

            y1new = 0
            x1new = int((y1new - y1) / slope + x1)
            # cv.line(board, (x1, y1), (x2, y2), color, 1)
            # cv.line(board, (x1, y1), (x2new, y2new), color, 1)
            cv.line(board, (x1new, y1new), (x2new, y2new), color, 1)
            intercept.append(x2new)
            vert.append([x1new, y1new, x2new, y2new])

imageshow(board)

from sklearn.cluster import KMeans

a = np.array(intercept).reshape(-1, 1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(a)
from operator import itemgetter
sorted(vert, key=itemgetter(2))

realLines = []
cur_group = []
i = 0
last_intercept = None
for v in vert:
    cur_group.append(i)
    if last_intercept is not None:
        if (intercept[i] - last_intercept) < 40:
            cur_group.append(intercept[i])
        else:
            print("Average", cur_group)
            realLines.append(np.average(cur_group))
            cur_group =[]
    else:
        cur_group.append(i)
    last_intercept = intercept[i]
    i = i + 1

np.argsort(vert,)