import cv2 as cv
import os
import random

target_path = 'images/target/lena.jpg'
input_path = ''

target_im = cv.imread(target_path)
input_files = []

input_files_size = 20

# resize
scale_percent = 40 # percent of original size
width = int(target_im.shape[1] * scale_percent / 100)
height = int(target_im.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv.resize(target_im, dim, interpolation = cv.INTER_AREA)
 

kernel_size = resized.shape[1]//input_files_size
print(kernel_size)

col = resized.shape[0]//kernel_size
row = resized.shape[1]//kernel_size
for i in range(row):
    for j in range(col):
        color = (random.random()*255, random.random()*255, random.random()*255)
        if i == 0:
            cv.rectangle(resized, (i*kernel_size, j*kernel_size), (kernel_size+(i*kernel_size), kernel_size+(j*kernel_size)), color, 1)
        else:
            cv.rectangle(resized, (1+(i*kernel_size), 1+(j*kernel_size)), (kernel_size+(i*kernel_size), kernel_size+(j*kernel_size)), color, 1)
cv.imshow('lena', resized)

cv.waitKey(0)
cv.destroyAllWindows()