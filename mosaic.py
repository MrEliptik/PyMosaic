import os
import random
import cv2 as cv
import numpy as np

import argparse

import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster

import imageio

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
    return tuple(peak)

def findBestColorMatch(target, dominant_colors):
    def distance(a, b):
        return pow(abs(a[0] - b[0]), 2) + pow(abs(a[1] - b[2]), 2)

    # closest value in the list
    return min(dominant_colors, key=lambda x: distance(x, target))

def visualizeDominantColors(target_im, images, resize_factor=1, keep_original=False):
    if resize_factor != 1:
        resized = resize(target_img, resize_factor)
    else:
        resized = target_img
    
    if keep_original:
        original = resized.copy()

    # Copy image to create the mosaic
    mosaic = np.zeros(resized.shape, np.uint8)

    #kernel_size = resized.shape[1]//len(input_imgs)
    kernel_size = resized.shape[1]//120

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
                cv.rectangle(mosaic, (x1, y1), (x2, y2), color, -1)         
            else:
                x1 = 1+(j*kernel_size)
                y1 = 1+(i*kernel_size)
                x2 = kernel_size+(j*kernel_size)
                y2 = kernel_size+(i*kernel_size)
                color = getDominantColor(resized[y1:y2, x1:x2,:])
                cv.rectangle(mosaic, (x1, y1), (x2, y2), color, -1)

    if keep_original:
        return mosaic, original
    else:
        return mosaic

def createMosaic(target_img, dominant_colors, images, repeat=True, resize_factor=1, keep_original=False):
    """ Recreate a target image as a mosaic with multiple images

    Args
        im_color_map : dictionnary {dominant_color: image}
        dominant_colors : list of dominant colors, same order as images
        images : list of images used to reproduce target_img
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
    else:
        resized = target_img
    
    if keep_original:
        original = resized.copy()

    # Copy image to create the mosaic
    mosaic = np.zeros(resized.shape, np.uint8)

    #kernel_size = resized.shape[1]//len(input_imgs)
    kernel_size = resized.shape[1]//120

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

                # Find the image that fits best the curr dominant color
                match = findBestColorMatch(color, dominant_colors)

                # Resize and put the image in the corresponding rectangle
                mosaic[y1:y2, x1:x2,:] = cv.resize(images[dominant_colors.index(match)], dsize=(x2-x1, y2-y1), interpolation=cv.INTER_CUBIC)
            else:
                x1 = 1+(j*kernel_size)
                y1 = 1+(i*kernel_size)
                x2 = kernel_size+(j*kernel_size)
                y2 = kernel_size+(i*kernel_size)
                color = getDominantColor(resized[y1:y2, x1:x2,:])

                # Find the image that fits best the curr dominant color
                match = findBestColorMatch(color, dominant_colors)

                # Resize and put the image in the corresponding rectangle
                mosaic[y1:y2, x1:x2,:] = cv.resize(images[dominant_colors.index(match)], dsize=(x2-x1, y2-y1), interpolation=cv.INTER_CUBIC)

    if keep_original:
        return mosaic, original
    else:
        return mosaic


def getDominantColors(images):
    dominant_colors = []
    for im in images:
        dominant_colors.append(getDominantColor(im))

    return dominant_colors 

def main(args):
    start = time.time()
    #target_path = 'images/target/bond.jpg'
    #input_path = 'images/input/'
    image_extensions = ('.png', '.jpg', '.jpeg', '.jfiff', '.tiff', '.bmp')
    input_files = []
    target_im = cv.imread(args.target_im)

    if args.grayscale:
        # Convert to grayscale and back to BGR to
        # keep the 3 channels
        target_im = cv.cvtColor(cv.cvtColor(target_im, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)

    for subdir, _, files in os.walk(args.inputs):
        print('[INFO] Working on: ' + str(subdir))
        for _file in files:
            if str(_file).lower().endswith(image_extensions):
                input_files.append(os.path.join(subdir, _file))

    print('[INFO] Found {} input files.'.format(len(input_files)))

    # Resize image to find clusters faster
    images = []
    for file in input_files:
        im = cv.imread(file)
        if args.grayscale:
            # TODO: is this the best way to do it ??
            images.append(cv.cvtColor(cv.cvtColor(resize(im, 0.2), cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR))
        else:
            images.append(resize(im, 0.2))

    # Create mapping for each image
    dominant_colors = getDominantColors(images)

    # Create the mosaic
    mosaic, original = createMosaic(target_im, dominant_colors, images, repeat=True, resize_factor=1, keep_original=True)

    print('[Info] Finished, took {} s'.format(time.time() - start))       

    cv.imshow('lena', original)
    cv.imshow('colors', mosaic)

    cv.waitKey(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Use many images to recreate a target image as a mosaic.')
    parser.add_argument('--target_im', type=str, required=True, help='Path to target image')
    parser.add_argument('--inputs', type=str, required=True, help='Path to input images')
    parser.add_argument('--resize_factor', type=float, default=1.0, help='Factor to resize target image')
    parser.add_argument('--grayscale', action='store_true', default=False, help='Convert to grayscale')
    parser.add_argument('--pixel_density', type=float, default=0.7, 
        help='Will effect number of images used to create the mosaic. 1 is 1 image per pixel, default=0.7')

    args = parser.parse_args()

    main(args)


