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

from multiprocessing.pool import Pool

import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()        
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result   
    return timed


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


def getDominantColors(images):
    dominant_colors = []
    for im in images:
        dominant_colors.append(getDominantColor(im))

    return dominant_colors 


def findBestColorMatch(target, dominant_colors):
    def distance(a, b):
        return pow(abs(a[0] - b[0]), 2) + pow(abs(a[1] - b[2]), 2)

    # closest value in the list
    return min(dominant_colors, key=lambda x: distance(x, target))


def createMosaic(target_img, dominant_colors, images, pixel_density=0.7, 
    repeat=True, resize_factor=1, keep_original=False, multithreading=True, num_workers=4):
    """ Recreate a target image as a mosaic with multiple images

    Args
        im_color_map : dictionnary {dominant_color: image}
        dominant_colors : list of dominant colors, same order as images
        images : list of images used to reproduce target_img
        pixel_density : value [0,1] use to calculate number of images to use
            to simulate pixels
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

    #kernel_i_size = resized.shape[0]*pixel_density
    #kernel_j_size = resized.shape[1]*pixel_density

    kernel_i_size = resized.shape[0]//100
    kernel_j_size = resized.shape[0]//100
    print(kernel_i_size, kernel_j_size)

    print('kernel size: {},{}'.format(kernel_i_size, kernel_j_size))

    colors = []
    patches = []

    # Gather the patches
    col = resized.shape[1]//kernel_j_size
    row = resized.shape[0]//kernel_i_size
    for i in range(row):
        for j in range(col):
            if i == 0:
                x1 = j*kernel_j_size
                y1 = i*kernel_i_size
                x2 = kernel_j_size+(j*kernel_j_size)
                y2 = kernel_i_size+(i*kernel_i_size)

                if args.multithreading: 
                    patches.append(resized[y1:y2, x1:x2,:])
                else:
                    color = getDominantColor(resized[y1:y2, x1:x2,:])
                    # Find the image that fits best the curr dominant color
                    match = findBestColorMatch(color, dominant_colors)
                    # Resize and put the image in the corresponding rectangle
                    mosaic[y1:y2, x1:x2,:] = cv.resize(images[dominant_colors.index(match)], 
                                                    dsize=(x2-x1, y2-y1), interpolation=cv.INTER_CUBIC)        
            else:
                x1 = 1+(j*kernel_j_size)
                y1 = 1+(i*kernel_i_size)
                x2 = kernel_j_size+(j*kernel_j_size)
                y2 = kernel_i_size+(i*kernel_i_size)

                if args.multithreading:
                    patches.append(resized[y1:y2, x1:x2,:])
                else:
                    color = getDominantColor(resized[y1:y2, x1:x2,:])

                    # Find the image that fits best the curr dominant color
                    match = findBestColorMatch(color, dominant_colors)

                    # Resize and put the image in the corresponding rectangle
                    mosaic[y1:y2, x1:x2,:] = cv.resize(images[dominant_colors.index(match)], 
                                                    dsize=(x2-x1, y2-y1), interpolation=cv.INTER_CUBIC)

    if args.multithreading:
        # Use thread to calculate dominant colors
        pool = Pool()
        # Create mapping for each image
        colors = pool.map(getDominantColor, patches) 
        pool.close() 
        pool.join()
    
        # Reconstruct image
        k = 0
        for i in range(row):
            for j in range(col):
                if i == 0:
                    x1 = j*kernel_j_size
                    y1 = i*kernel_i_size
                    x2 = kernel_j_size+(j*kernel_j_size)
                    y2 = kernel_i_size+(i*kernel_i_size)
                else:
                    x1 = 1+(j*kernel_j_size)
                    y1 = 1+(i*kernel_i_size)
                    x2 = kernel_j_size+(j*kernel_j_size)
                    y2 = kernel_i_size+(i*kernel_i_size)

                match = findBestColorMatch(colors[k], dominant_colors)

                mosaic[y1:y2, x1:x2,:] = cv.resize(images[dominant_colors.index(match)], 
                                                    dsize=(x2-x1, y2-y1), interpolation=cv.INTER_CUBIC)
                k += 1

    if keep_original:
        return mosaic, original
    else:
        return mosaic

def main(args):
    start = time.time()
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

    if args.multithreading:
        # Create a pool of thread 
        # (same as number of cores)
        pool = Pool(args.num_workers)
        # Create mapping for each image
        dominant_colors = pool.map(getDominantColor, images) 
        pool.close() 
        pool.join()
    else:
        # Create mapping for each image (single thread)
        dominant_colors = getDominantColors(images)

    # Create the mosaic
    mosaic, original = createMosaic(target_im, dominant_colors, images, 
        pixel_density=args.pixel_density, repeat=True, resize_factor=1, 
        keep_original=True, multithreading=args.multithreading, num_workers=args.num_workers)

    print('[Info] Finished, took {} s'.format(time.time() - start))       

    cv.imshow(os.path.basename(args.target_im), original)
    cv.imshow('mosaic', mosaic)

    cv.waitKey(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Use many images to recreate a target image as a mosaic.')
    parser.add_argument('--target_im', type=str, required=True, help='Path to target image')
    parser.add_argument('--inputs', type=str, required=True, help='Path to input images')
    parser.add_argument('--resize_factor', type=float, default=1.0, help='Factor to resize target image')
    parser.add_argument('--grayscale', action='store_true', default=False, help='Convert to grayscale')
    parser.add_argument('--pixel_density', type=float, default=0.7, 
        help='Will effect number of images used to create the mosaic. 1 is 1 image per pixel, default=0.7')
    parser.add_argument('--multithreading', action='store_true', default=False, 
        help='Use multiple thread to create the mosaic')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers to use in multithreading')

    args = parser.parse_args()

    main(args)


