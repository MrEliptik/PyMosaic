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

def resize(image, factor):
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)
    dim = (width, height)
    # resize image
    if factor > 1:
        return cv.resize(image, dim, interpolation = cv.INTER_CUBIC)
    else:
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

def distance(a, b):
    return pow(abs(a[0] - b[0]), 2) + pow(abs(a[1] - b[2]), 2)

def colorDistance(a, b): 
    return (distance(a[0], b[0]), distance(a[1], b[1]), distance(a[2], b[2]))

def findBestColorMatch(target, dominant_colors):
    # closest value in the list
    return min(dominant_colors, key=lambda x: distance(x, target))

def autoContrast(im):
    # Converting image to LAB Color model 
    lab = cv.cvtColor(im, cv.COLOR_BGR2LAB)

    # Splitting the LAB image to different channels
    l, a, b = cv.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl,a,b))

    # Converting image from LAB Color model to RGB model
    return cv.cvtColor(limg, cv.COLOR_LAB2BGR)

def verify_alpha_channel(frame):
    try:
        frame.shape[3] # 4th position
    except IndexError:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
    return frame

def apply_color_overlay(frame, intensity=0.2, b = 0,g = 0,r = 0):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    color_bgra = (b, g, r, 1)
    overlay = np.full((frame_h, frame_w, 4), color_bgra, dtype='uint8')
    cv.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
    return frame

def createMosaic(target_img, dominant_colors, images, pixel_density=0.7, 
    repeat=True, resize_factor=1, keep_original=False, multithreading=True, 
    num_workers=4, output_size_factor=2):
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
    print('Original size: {}'.format(target_img.shape))
    if resize_factor != 1:
        resized = resize(target_img, resize_factor)
    else:
        resized = target_img
    
    if keep_original:
        original = resized.copy()

    print('Resized size: {}'.format(resized.shape))

    h,w,c = resized.shape

    # Copy image to create the mosaic
    mosaic = np.zeros((int(h*output_size_factor), int(w*output_size_factor), c), np.uint8)

    kernel_i_size = int(1//pixel_density)
    kernel_j_size = int(1//pixel_density)

    kernel_i_mosaic = int(kernel_i_size*output_size_factor)
    kernel_j_mosaic = int(kernel_j_size*output_size_factor)

    print('Kernel size: {},{}'.format(kernel_i_size, kernel_j_size))
    print('Kernel mosaic size: {},{}'.format(kernel_i_mosaic, kernel_j_mosaic))

    colors = []
    patches = []

    # Gather the patches
    col = resized.shape[1]//kernel_j_size
    row = resized.shape[0]//kernel_i_size
    for i in range(row):
        for j in range(col):
            x1 = j*kernel_j_size
            y1 = i*kernel_i_size
            x2 = kernel_j_size+(j*kernel_j_size)
            y2 = kernel_i_size+(i*kernel_i_size)

            x1_mosaic = j*kernel_j_mosaic
            y1_mosaic = i*kernel_i_mosaic
            x2_mosaic = kernel_j_mosaic+(j*kernel_j_mosaic)
            y2_mosaic = kernel_i_mosaic+(i*kernel_i_mosaic)

            if args.multithreading:
                patches.append(resized[y1:y2, x1:x2,:])
            else:
                color = getDominantColor(resized[y1:y2, x1:x2,:])

                # Find the image that fits best the curr dominant color
                match = findBestColorMatch(color, dominant_colors)
                    
                # Get the image
                im_match = images[dominant_colors.index(match)]

                if color_filter:
                    delta = colorDistance(match, color)
                    print(delta)
                    cv.imshow('image match', im_match)
                    im_match = apply_color_overlay(im_match, intensity=0.6, b=delta[0], g=delta[1], r=delta[2])
                    cv.imshow('match after filtering', im_match)
                    color_square = np.zeros(resized.shape, np.uint8)
                    color_square[:] = color
                    cv.imshow('color', color_square)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                # Resize and put the image in the corresponding rectangle
                mosaic[y1_mosaic:y2_mosaic, x1_mosaic:x2_mosaic,:] = cv.resize(images[dominant_colors.index(match)], 
                                                dsize=(x2_mosaic-x1_mosaic, y2_mosaic-y1_mosaic), interpolation=cv.INTER_CUBIC)

    if args.multithreading:
        # Use thread to calculate dominant colors
        pool = Pool(num_workers)
        # Create mapping for each image
        colors = pool.map(getDominantColor, patches) 
        pool.close() 
        pool.join()
    
        # Reconstruct image
        k = 0
        for i in range(row):
            for j in range(col):
                x1 = j*kernel_j_size
                y1 = i*kernel_i_size
                x2 = kernel_j_size+(j*kernel_j_size)
                y2 = kernel_i_size+(i*kernel_i_size)

                x1_mosaic = (j*kernel_j_mosaic)
                y1_mosaic = (i*kernel_i_mosaic)
                x2_mosaic = kernel_j_mosaic+(j*kernel_j_mosaic)
                y2_mosaic = kernel_i_mosaic+(i*kernel_i_mosaic)

                match = findBestColorMatch(colors[k], dominant_colors)

                mosaic[y1_mosaic:y2_mosaic, x1_mosaic:x2_mosaic,:] = cv.resize(images[dominant_colors.index(match)], 
                                                dsize=(x2_mosaic-x1_mosaic, y2_mosaic-y1_mosaic), interpolation=cv.INTER_CUBIC)
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

    if args.contrast:
        target_im = autoContrast(target_im)

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
        pixel_density=args.pixel_density, repeat=True, 
        resize_factor=args.resize_factor, keep_original=True, 
        multithreading=args.multithreading, num_workers=args.num_workers, 
        output_size_factor=args.output_size_factor, color_filter=args.color_filter)

    print('[Info] Finished, took {} s'.format(time.time() - start))    

    if args.save:
        cv.imwrite(
            os.path.join('results', os.path.basename(args.target_im) + '_mosaic.jpg'),
            mosaic)   

    cv.imshow(os.path.basename(args.target_im), original)
    cv.imshow('mosaic', mosaic)

    cv.waitKey(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Use many images to recreate a target image as a mosaic.')
    parser.add_argument('--target_im', type=str, required=True, help='Path to target image')
    parser.add_argument('--inputs', type=str, required=True, help='Path to input images')
    parser.add_argument('--output_size_factor', type=float, help='How much times the output should be bigger than the target')
    parser.add_argument('--resize_factor', type=float, default=1.0, help='Factor to resize target image')
    parser.add_argument('--grayscale', action='store_true', default=False, help='Convert to grayscale')
    parser.add_argument('--pixel_density', type=float, default=0.7, 
        help='Will effect number of images used to create the mosaic. 1 is 1 image per pixel, default=0.7')
    parser.add_argument('--multithreading', action='store_true', default=False, 
        help='Use multiple thread to create the mosaic')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers to use in multithreading')
    parser.add_argument('--save', action='store_true', default=False, help='Save output mosaic')
    parser.add_argument('--show', action='store_true', default=False, help='Show output mosaic')
    parser.add_argument('--contrast', action='store_true', default=False, help='Apply auto contrast to target image')
    parser.add_argument('--color_filter', action='store_true', default=False, help='Apply color filters to get closer to the desired color')

    args = parser.parse_args()

    main(args)


