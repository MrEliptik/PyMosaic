import cv2
import numpy as np
from mosaic import resize
from mosaic import getDominantColor
from mosaic import verify_alpha_channel
from mosaic import apply_color_overlay

def nothing(x):
    pass

# Create a black image, a window
img = cv2.imread('images/target/bond.jpg')
img = resize(img, 0.4)
cv2.namedWindow('image')

color = getDominantColor(img)

switch = '0 : OFF \n1 : ON'

# create trackbars for color change
cv2.createTrackbar('R','image', int(color[0]), 255,nothing)
cv2.createTrackbar('G','image', int(color[1]), 255,nothing)
cv2.createTrackbar('B','image', int(color[2]), 255,nothing)

cv2.createTrackbar('Intensity','image', 30, 100,nothing)

# create switch for ON/OFF functionality
cv2.createTrackbar(switch, 'image',1,1,nothing)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    todisplay = img.copy()

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    i = cv2.getTrackbarPos('Intensity', 'image')/100

    s = cv2.getTrackbarPos(switch,'image')

    if s:
        todisplay = apply_color_overlay(img.copy(), i, b, g, r)   
    else:
        todisplay = img.copy()
 
    cv2.imshow('image',todisplay)

cv2.destroyAllWindows()