import numpy as np
import cv2
import imutils

def get_symbols(image, debug=False):
    #res = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(image, (1000, 1000), None)

    _,thresh = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,5), np.uint8)
    dial = cv2.dilate(thresh, kernel, iterations=2)

    params = cv2.SimpleBlobDetector_Params()

    params.blobColor = 255
    params.filterByColor = True

    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 100000
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByCircularity = False

    detector = cv2.SimpleBlobDetector(params)

    rev = (255-dial)

    holes = detector.detect(rev)

    for h in holes:
        x, y = h.pt
        s = int(h.size * 1.2)

        cv2.circle(dial, (int(x), int(y)), s, (255,255,255), -1)

    keypoints = detector.detect(dial)

    if debug:
        small = cv2.resize(dial, (600,600), None)
        cv2.imshow('a', small)
        cv2.waitKey(0)

    symb = []

    for k in keypoints:
        x, y = k.pt
        s = int(k.size * 1.2) # ~ sqrt(2) / 2

        crop = res[int(max(y-s,0)):int(min(y+s,1000)), int(max(x-s,0)):int(min(x+s,1000))]

        crop_res = cv2.resize(crop, (100, 100), None)

        symb.append(crop_res)

    return (symb, keypoints)
