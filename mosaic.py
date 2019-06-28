import os
import random
import cv2 as cv
import numpy as np

import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster

import time

def resize(image, factor):
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)
    dim = (width, height)
    # resize image
    return cv.resize(image, dim, interpolation = cv.INTER_AREA)

def getDominantColor(image, num_clusters=5):
    shape = image.shape
    ar = image.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, num_clusters)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]

    return peak

def createMosaic(target_img, input_imgs, repeat=True, resize_factor=1, keep_original=False):
    """ Recreate a target image as a mosaic with multiple images

    Args
        target_img : image to reproduce
        input_imgs : images used to reproduce target_img
        repeat : use multiple time input images, default=True
        resize_factor : resize factor for target_img, default=1 (no resize)
        keep_original : keep the original and returns it

    Returns
        mosaic            : the mosaic image

        or

        mosaic, original : the mosaic image, the original image
    """
    if resize_factor != 1:
        resized = resize(target_img, resize_factor)
    
    if keep_original:
        original = resized.copy()

    #kernel_size = resized.shape[1]//len(input_imgs)
    kernel_size = resized.shape[1]//100

    col = resized.shape[1]//kernel_size
    row = resized.shape[0]//kernel_size
    for i in range(row):
        for j in range(col):
            if i == 0:
                x1 = j*kernel_size
                y1 = i*kernel_size
                x2 = kernel_size+(j*kernel_size)
                y2 = kernel_size+(i*kernel_size)
                color = getDominantColor(resized[y1:y2, x1:x2,:])
                cv.rectangle(resized, (x1, y1), (x2, y2), color, -1)         
            else:
                x1 = 1+(j*kernel_size)
                y1 = 1+(i*kernel_size)
                x2 = kernel_size+(j*kernel_size)
                y2 = kernel_size+(i*kernel_size)
                color = getDominantColor(resized[y1:y2, x1:x2,:])
                cv.rectangle(resized, (x1, y1), (x2, y2), color, -1)
    if keep_original:
        return resized, original
    else:
        return resized

def main():
    target_path = 'images/target/lena.jpg'
    input_path = 'images/input/'
    image_extensions = ('.png', '.jpg', '.jpeg', '.jfiff', '.tiff', '.bmp')
    input_files = []

    target_im = cv.imread(target_path)

    for subdir, _, files in os.walk(input_path):
        print('[INFO] Working on: ' + str(subdir))
        for _file in files:
            if str(_file).lower().endswith(image_extensions):
                input_files.append(os.path.join(subdir, _file))

    print('[INFO] Found {} input files.'.format(len(input_files)))

    mosaic, original = createMosaic(target_im, input_files, repeat=True, resize_factor=0.30, keep_original=True)
            
    cv.imshow('lena', original)
    cv.imshow('colors', mosaic)

    cv.waitKey(0)

if __name__ == "__main__":
    main()



