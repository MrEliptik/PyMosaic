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

def getDominantColor(image, num_clusters=5):
    shape = image.shape
    ar = image.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, num_clusters)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]

    return peak

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

input_files_size = len(input_files)

# TODO: Remove, just for testing
input_files_size = 100
print('[INFO] Found {} input files.'.format(input_files_size))

# resize params
scale_percent = 35 # percent of original size
width = int(target_im.shape[1] * scale_percent / 100)
height = int(target_im.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv.resize(target_im, dim, interpolation = cv.INTER_AREA)
original = resized.copy()

kernel_size = resized.shape[1]//input_files_size

col = resized.shape[1]//kernel_size
row = resized.shape[0]//kernel_size
for i in range(row):
    for j in range(col):
        #color = (random.random()*255, random.random()*255, random.random()*255)
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
        
cv.imshow('lena', original)
cv.imshow('colors', resized)

cv.waitKey(0)

