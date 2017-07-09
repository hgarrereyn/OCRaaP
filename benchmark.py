
import timeit


import model
import image_parser
import cv2
import numpy as np

model = model.Model(fromCheckpoint='milestones/classifier_v1.0.ckpt')

def load_and_parse():
    # read the image
    prog = cv2.imread("examples/helloWorld.jpg", cv2.IMREAD_GRAYSCALE)

    # fetch & predict symbols
    symbols, keypoints = image_parser.get_symbols(prog)
    raw = model.run_predict(symbols)
    predictions = np.argmax(raw[0], 1)

    return predictions


count = 1000
t = timeit.timeit(stmt="load_and_parse()", setup="from __main__ import load_and_parse", number=count) / count

print(str(t))
