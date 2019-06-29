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

import imageio

from multiprocessing.pool import ThreadPool

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

@timeit
def resize(image, factor):
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)
    dim = (width, height)
    # resize image
    return cv.resize(image, dim, interpolation = cv.INTER_AREA)

@timeit
def getDominantColor(image, num_clusters=5):
    shape = image.shape
    ar = image.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, num_clusters)


    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    return tuple(peak)

@timeit
def findBestColorMatch(target, dominant_colors):
    def distance(a, b):
        return pow(abs(a[0] - b[0]), 2) + pow(abs(a[1] - b[2]), 2)

    # closest value in the list
    return min(dominant_colors, key=lambda x: distance(x, target))

@timeit
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

@timeit
def getDominantColors(images):
    dominant_colors = []
    for im in images:
        dominant_colors.append(getDominantColor(im))

    return dominant_colors 

@timeit
def main():
    start = time.time()
    target_path = 'images/target/bond.jpg'
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

    # Resize image to find clusters faster
    images = []
    for file in input_files:
        im = cv.imread(file)
        images.append(resize(im, 0.2))

    # Create a pool of thread 
    # s(same as number of cores)
    pool = ThreadPool()
    # Create mapping for each image
    dominant_colors = pool.map(getDominantColor, images) 
    pool.close() 
    pool.join()
    # Create mapping for each image (single thread)
    #dominant_colors = getDominantColors(images)

    # Create the mosaic
    mosaic, original = createMosaic(target_im, dominant_colors, images, repeat=True, resize_factor=0.4, keep_original=True)

    print('[Info] Finished, took {} s'.format(time.time() - start))       

    cv.imshow(os.path.basename(target_path  ), original)
    cv.imshow('mosaic', mosaic)

    cv.waitKey(0)

if __name__ == "__main__":
    main()


