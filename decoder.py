import numpy as np
import cv2
import image_parser
import model

import time

s = time.time()
model = model.Model(fromCheckpoint='milestones/classifier_v1.0.ckpt')
e = time.time()

print(str(e-s) + ' seconds to load model')

image = cv2.imread('decode_tests/cat.jpeg')
image = cv2.resize(image, (1000,1000), None)

image_g = cv2.imread('decode_tests/cat.jpeg', cv2.IMREAD_GRAYSCALE)
symbols, keypoints = image_parser.get_symbols(image_g)

s = time.time()
out = model.run_predict(symbols)
e = time.time()

print(str(e-s) + ' seconds to predict')

predictions = np.argmax(out[0], 1)

for i in range(len(symbols)):
    cv2.imshow('a', symbols[i])

    k = keypoints[i]

    print(k.pt)

    c = (255,0,0)

    cl = ['sad', 'dead', 'at', 'hash', 'conf', 'empty', 'dot', 'dollar', 'plus', 'dash'][predictions[i]]

    cv2.circle(image, (int(k.pt[0]), int(k.pt[1])), int(k.size*1.2), c, 2)
    cv2.putText(image, cl, (int(k.pt[0]), int(k.pt[1]) + int(k.size)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    print(cl)

    #cv2.waitKey(0)

small = cv2.resize(image, (600,600), None)

cv2.imshow('a',small)
cv2.waitKey(0)
