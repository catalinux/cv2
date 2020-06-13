import numpy as np
import cv2 as cv
import imutils
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import pickle

from proiect.lib.util import imageshow, auto_canny

greenLower = (40, 60, 100)
greenUpper = (80, 255, 255)
aprox_param = 0.035


def clahe(bgr, gridsize=11):
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    lab_planes = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv.merge(lab_planes)
    bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    return bgr


def cluster(image, K=4):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    # define criter§ia, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    # imageshow(res2, "quant")
    img = cv.cvtColor(res2.astype(np.uint8), cv.COLOR_BGR2RGB)
    # imageshow(img)
    return res2, label


def circles(orig, table):
    orig = cv.GaussianBlur(orig, (5, 5), 0)
    edges = cv.Canny(orig, threshold1=50, threshold2=60)
    imageshow(edges, "edge circles")
    # imageshow(orig)
    # orig = auto_canny(orig)
    # imageshow(orig)
    diameterBall = 16
    radiusBall = int(round(diameterBall / 2))
    gray = edges
    imageshow(gray, "Detect circles")
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, diameterBall * 0.7,
                              minRadius=int(round(radiusBall * 0.75)),
                              maxRadius=int(round(radiusBall * 1.25)),
                              param1=100,
                              param2=10
                              )
    print(circles)
    board = table.copy()
    output = gray
    # ensure at least some circles were found
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
        # show the output image
        # cv.imshow("output", np.hstack([image, output]))
        # cv.waitKey(0)
        imageshow(np.hstack([table, board]))
        imageshow(edges)
    #


def detect_balls_using_hsv(table):
    low_white = np.array([80, 140, 65])
    high_white = np.array([255, 255, 255])
    mask_white = cv.inRange(table, low_white, high_white)
    white_ball = cv.bitwise_and(table, table, mask=mask_white)

    low_red = np.array([0, 0, 26])
    high_red = np.array([26, 42, 255])
    mask_red = cv.inRange(table, low_red, high_red)
    red_ball = cv.bitwise_and(table, table, mask=mask_red)

    low_blue = np.array([91, 0, 0])
    high_blue = np.array([180, 135, 55])
    mask_blue = cv.inRange(table, low_blue, high_blue)
    blue_ball = cv.bitwise_and(table, table, mask=mask_blue)

    low_pink = np.array([49, 0, 187])
    high_pink = np.array([255, 162, 255])
    mask_pink = cv.inRange(table, low_pink, high_pink)
    pink_ball = cv.bitwise_and(table, table, mask=mask_pink)

    low_dark_green = np.array([30, 23, 0])
    high_dark_green = np.array([90, 255, 30])
    mask_dark_green = cv.inRange(table, low_dark_green, high_dark_green)

    low_brown = np.array([0, 52, 37])
    high_brown = np.array([45, 105, 166])
    mask_brown = cv.inRange(table, low_brown, high_brown)

    low_yellow = np.array([0, 107, 68])
    high_yellow = np.array([114, 255, 255])
    mask_yellow = cv.inRange(table, low_yellow, high_yellow)

    low_black = np.array([0, 0, 0])
    high_black = np.array([45, 38, 65])
    mask_black = cv.inRange(table, low_black, high_black)

    mask_balls = mask_white + mask_red + mask_blue + mask_pink + mask_dark_green + mask_brown + mask_yellow + mask_black
    mask_balls = cv.erode(mask_balls, None, iterations=1)
    mask_balls = cv.dilate(mask_balls, None, iterations=1)

    balls = cv.bitwise_and(table, table, mask=mask_balls)
    imageshow(table, "table")
    imageshow(balls, "balls"),
    imageshow(mask_balls, "mask balls")

    mask_balls = cv.GaussianBlur(mask_balls, (13, 13), 0)  # use gaussian blur
    imageshow(mask_balls, "mask balls gauss")

    return table, balls, mask_balls, 2


maxBalls = {
    'blue': 1,
    'pink': 1,
    'yellow': 1,
    'green': 1,
    'brown': 1,
    'black': 1,
    'white': 1,
    'red': 15
}

colors = {
    'blue': [
        np.array([91, 0, 0]),
        np.array([180, 135, 55])
    ],
    'yellow': [
        np.array([0, 107, 68]),
        np.array([114, 255, 255])
    ],
    'pink': [
        np.array([49, 0, 187]),
        np.array([255, 162, 255])
    ],
    'green': [
        np.array([30, 23, 0]),
        np.array([90, 255, 30])
    ],
    'brown': [np.array([0, 52, 37]), np.array([45, 105, 166])],
    'black': [np.array([0, 0, 0]), np.array([45, 38, 65])],
    'white': [np.array([0, 0, 240]), np.array([255, 137, 255])],
    'red': [np.array([0, 0, 26]), np.array([26, 42, 255])]
}


def get_balls(image, color):
    lower = colors[color][0]
    upper = colors[color][1]
    print("range:", lower, upper)
    mask = cv.inRange(image, lower, upper)
    mask = cv.erode(mask, None, iterations=1)
    mask = cv.dilate(mask, None, iterations=1)

    res = cv.bitwise_and(image, image, mask=mask)
    imageshow(res, "get balls" + color)
    gray = cv.cvtColor(res.copy(), cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)  # use gaussian blur
    edges = cv.Canny(gray, 10, 150, apertureSize=3)
    cnts, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    circles = []
    for a in cnts:
        (x, y), radius = cv.minEnclosingCircle(a)
        if radius > 4 and radius < 10:
            print(radius)
            circles.append(a)

    diameterBall = 16
    radiusBall = int(round(diameterBall / 2))
    circles2 = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, diameterBall * 0.7,
                              minRadius=int(round(radiusBall * 0.25)),
                              maxRadius=int(round(radiusBall * 1.65)),
                              param1=100,
                              param2=10
                              )
    print(circles2)
    if circles2 is not None:
        circles2 = np.uint16(np.around(circles2))
        for i in circles2[0, :]:
            # draw the outer circle
            cv.circle(res, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(res, (i[0], i[1]), 2, (0, 0, 255), 3)

    # cv.drawContours(res, circles, -1, (233, 0, 0), 3)
    imageshow(res, "cont")
    imageshow(edges)


def spec(orig):
    gray = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 110, 220, cv.THRESH_BINARY)[1]
    # thresh = cv.adaptiveThreshold(gray, 140, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    imageshow(thresh, "spec")
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = contours
    tt = orig.copy()
    hull_list = []
    circles = []
    for i in range(len(cnts)):
        area = cv.contourArea(cnts[i])
        if area < 1 or area > 200:
            continue
        hull = cv.convexHull(cnts[i])
        print(area)
        hull_list.append(hull)
        (x, y), radius = cv.minEnclosingCircle(cnts[i])
        if radius > 4:
            continue
        print(x, y, radius)
        circles.append([(x, y), radius])
        radius = int(radius)
        cv.circle(tt, (int(x), int(y)), radius, (200, 200, 120), 2)

    print(circles)
    cv.drawContours(tt, hull_list, -1, (244, 100, 0), 3)
    imageshow(tt)
    pass


def lines(orig):
    gray = cv.cvtColor(orig.copy(), cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)  # use gaussian blur
    edges = cv.Canny(gray, 10, 150, apertureSize=3)
    imageshow(edges)

    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=250)

    print(lines)
    imageshow(orig, "lines")
    for x1, y1, x2, y2 in lines[0]:
        # cv.drawContours()
        cv.line(orig, (x1, y1), (x2, y2), (233, 255, 0), 4)
    imageshow(orig, "Lines")
    pass


# lines(t)
# lines(t)
def w(orig):
    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([0, 0, 255], dtype=np.uint8)
    mask = cv.inRange(orig, lower_white, upper_white)
    res = cv.bitwise_and(orig, orig, mask=mask)
    imageshow(res, "res")
    shifted = cv.pyrMeanShiftFiltering(orig, 21, 51)
    imageshow(shifted)
    gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255,
                          cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    imageshow(thresh, "w")
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
                              labels=thresh)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv.minEnclosingCircle(c)
        cv.circle(orig, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv.putText(orig, "#{}".format(label), (int(x) - 10, int(y)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        imageshow(orig)
    pass


def do_it(filename):
    print(filename)
    im = cv.imread(filename)
    im = imutils.resize(im, 1200)
    orig = im.copy()
    # imageshow(im)
    # im = cluster(im)
    # imageshow(im,"before blurr")
    im = cv.GaussianBlur(im, (13,   13), 0)
    # imageshow(np.hstack([blurred,im]))
    # blurred = auto_canny(blurred)
    # imageshow(blurred, "blurred")
    hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    # hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])
    # imageshow(hsv, "hsv")
    mask = cv.inRange(hsv, greenLower, greenUpper)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(contours)
    cnts = contours
    # imageshow(im)
    # aprox_param = 0.015
    rectangles = []
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    cv.drawContours(im, cnts, -1, (0, 255, 0), 3)
    hull_list = []
    for i in range(len(cnts)):
        hull = cv.convexHull(cnts[i])
        hull_list.append(hull)
    cv.drawContours(im, hull_list, 0, (0, 255, 0), 3)
    pickle_out = open("dict.pickle", "wb")
    pickle.dump(hull_list[0], pickle_out)
    pickle_out.close()

    img = im.copy()
    kmeans = KMeans(n_clusters=4, random_state=0).fit(cnts[0][:, 0])
    cv.circle(img, tuple(kmeans.cluster_centers_[0].astype(int)), 4, (200, 0, 0), 3)
    cv.circle(img, tuple(kmeans.cluster_centers_[1].astype(int)), 4, (200, 0, 0), 3)
    cv.circle(img, tuple(kmeans.cluster_centers_[2].astype(int)), 4, (200, 0, 0), 3)
    cv.circle(img, tuple(kmeans.cluster_centers_[3].astype(int)), 4, (200, 0, 0), 3)
    imageshow(img,"cluster")

    # imageshow(im, "hull")
    # if len(cnts) > 0:
    #     cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    #     for c in cnts:
    #         # approximate the contour
    #         peri = cv.arcLength(c, True)
    #         approx = cv.approxPolyDP(c, aprox_param * peri, True)
    #         if len(approx) == 4:
    #             contourArea = cv.contourArea(c)
    #             # print("Area ", contourArea)
    #             rectangles.append(approx)
    cv.drawContours(im, hull_list, 0, (0, 0, 255), 3)
    imageshow(im)
    # imageshow(mask)
    stencil = np.zeros(im.shape[:-1]).astype(np.uint8)
    mask_value = 255
    cv.fillPoly(stencil, hull_list[:1], mask_value)
    # imageshow(stencil)
    fill_color = [0, 0, 0]
    sel = stencil != mask_value
    orig[sel] = fill_color
    imageshow(orig)

    # w(orig)
    # orig_b = cv.GaussianBlur(orig, (125, 125), 0)
    # imageshow(orig_b)
    # imageshow(orig-orig_b)
    # orig = cv.GaussianBlur(orig, (15, 15), 0)
    # imageshow(orig)
    # circles(orig)
    # spec(orig)
    # table, balls, mask_balls, au = detect_balls_using_hsv(orig)
    table = orig
    # # contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cnts = contours
    # tt = orig.copy()
    # hull_list = []
    # circles = []
    # for i in range(len(cnts)):
    #     area = cv.contourArea(cnts[i])
    #     if area < 10 or area > 200:
    #         continue
    #     hull = cv.convexHull(cnts[i])
    #     print(area)
    #     hull_list.append(hull)
    #     (x, y), radius = cv.minEnclosingCircle(cnts[i])
    #     circles.append([(x, y), radius])
    #
    # print(circles)
    # c_table = cluster(table, 10)
    # imageshow(c_table, "clustered table")

    # lines(table)
    # circles(balls,table)
    # circles(au, table)
    # circles(mask_balls, table)
    # get_balls(table, 'blue')
    # get_balls(table, 'pink')
    # get_balls(table, 'brown')
    # get_balls(table, 'yellow')
    # get_balls(table, 'green')
    # get_balls(table, 'black')
    # get_balls(table, 'red')
    get_balls(table, 'white')
    return table


filename = '49.jpg'
filename = '24.jpg'
# filename = False
import logging

import os


def process_table(t):
    t_orig = t.copy()
    cv.imwrite('table.jpg', t)
    t = cv.GaussianBlur(t, (15, 15), 0)
    t = clahe(t)
    a, l = cluster(t, 10)
    stats = np.unique(l, return_counts=True)

    to_remove = stats[0][stats[1] > 10000]
    imageshow(np.hstack([a, t,t_orig]))
    aaa = np.in1d(l.flatten(), to_remove)
    aaa.resize(675, 1200)
    tt = t_orig.copy()
    tt[aaa] = 0

    imageshow(tt, "remove green")
    gray = cv.cvtColor(tt, cv.COLOR_BGR2GRAY)
    # thresh = cv.threshold(gray, 110, 220, cv.THRESH_BINARY)[1]
    thresh = cv.adaptiveThreshold(gray, 140, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    # imageshow(thresh, "spec")
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = contours
    cv.drawContours(thresh, cnts, -1, (0, 222, 0), 2)
    imageshow(thresh, "thres")
    # spec(tt)

    gray = cv.cvtColor(tt, cv.COLOR_BGR2GRAY)
    diameterBall = 16
    radiusBall = int(round(diameterBall / 2))
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, diameterBall * 0.7,
                              minRadius=int(round(radiusBall * 0.25)),
                              maxRadius=int(round(radiusBall * 1.65)),
                              param1=100,
                              param2=10
                              )

    board = tt.copy()
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
        # show the output image
        # cv.imshow("output", np.hstack([image, output]))
        # cv.waitKey(0)
        # imageshow(np.hstack([board,t]))


if filename:
    t = do_it("data/training_data/Task1/" + filename)
    tt = process_table(t)
else:
    for file in os.listdir("data/training_data/Task1/"):
        if file.endswith(".jpg"):
            print(file)
            # try:
            t = do_it("data/training_data/Task1/" + file)
            tt = process_table(t)
            # except ValueError:
            #     print("Error on ", file, ValueError)
            #     logging.error("Error on " + file)
            # except AttributeError:
            #     print("Attribute Error on: " + AttributeError + file)
            #     logging.error("Error on " + file)


def thresh(img, conservative=0, min_blob_size=50):
  '''
    Get threshold to make mask using the otsus method, and apply a correction
    passed in conservative (-100;100) as a percentage of th.
  '''

  # blur and get level using otsus
  blur = cv.GaussianBlur(img, (13, 13), 0)
  level, _ = cv.threshold(
      blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)

  # print("Otsus Level: ",level)

  # change with conservative
  level += conservative / 100.0 * level

  # check boundaries
  level = 255 if level > 255 else level
  level = 0 if level < 0 else level

  # mask image
  _, mask = cv.threshold(blur, level, 255, cv.THRESH_BINARY)

  # morph operators
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
  mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
  mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

  # remove small connected blobs
  # find connected components
  n_components, output, stats, centroids = cv.connectedComponentsWithStats(
      mask, connectivity=8)
  # remove background class
  sizes = stats[1:, -1]
  n_components = n_components - 1

  # remove blobs
  mask_clean = np.zeros((output.shape))
  # for every component in the image, keep it only if it's above min_blob_size
  for i in range(0, n_components):
    if sizes[i] >= min_blob_size:
      mask_clean[output == i + 1] = 255

  return mask_clean 