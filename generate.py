#!/usr/local/bin/python

import numpy as np
import cv2
import imutils
import image_parser

import sys
import getopt

def get_convolutions(img):

    conv = []

    inv = (255-img)

    # rotations
    for angle in range(0,360,15):
        rotinv = imutils.rotate(inv, angle)
        rot = (255-rotinv)

        conv.append(rot)

    return conv


def help():
    print('generate.py -i [input image] -o [output folder]')

def run(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:o:v")
    except getopt.GetoptError:
        help()
        sys.exit(1)

    image_file = None
    out_path = None
    verify = False

    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        elif opt == '-i':
            image_file = arg
        elif opt == '-o':
            out_path = arg
            if out_path[len(out_path) - 1] != '/':
                out_path += '/'
        elif opt == '-v':
            verify = True

    if image_file == None or (out_path == None and verify == False):
        help()
        sys.exit(2)

    # Do the thing
    print('Scanning: ' + image_file)

    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    symb,_ = image_parser.get_symbols(image)

    print('Found ' + str(len(symb)) + ' symbols')

    if not verify:
        print('Writing to: ' + str(out_path))

    i = 0;

    for sym in symb:
        valid = True

        if verify:
            cv2.imshow('Verify', sym)
            print('Valid? (Y/n)')

            a = cv2.waitKey(0)

            if (chr(a & 0xff) == 'n'):
                print('-- Won\'t write')
                valid = False

        if valid:
            convs = get_convolutions(sym)

            for conv in convs:
                f = out_path + (str(i).zfill(4) + '.jpg')
                cv2.imwrite(f, conv)

                i += 1


if __name__ == '__main__':
    run(sys.argv[1:])
