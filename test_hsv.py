import cv2
import numpy as np
from mosaic import resize
from mosaic import getDominantColor

def nothing(x):
    pass

# Create a black image, a window
img = cv2.imread('images/target/bond.jpg')
img = resize(img, 0.4)
cv2.namedWindow('image')

color = getDominantColor(img)

switch = '0 : OFF \n1 : ON'

# create trackbars for color change
cv2.createTrackbar('Hue filter','image', 100, 500,nothing)
cv2.createTrackbar('Brightness filter','image', 100, 100,nothing)

cv2.createTrackbar('R','image', int(color[0]), 255,nothing)
cv2.createTrackbar('G','image', int(color[1]), 255,nothing)
cv2.createTrackbar('B','image', int(color[2]), 255,nothing)

# create switch for ON/OFF functionality
cv2.createTrackbar(switch, 'image',1,1,nothing)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    todisplay = img.copy()

    # get current positions of four trackbars
    hue = cv2.getTrackbarPos('Hue filter','image')/100
    brightness = cv2.getTrackbarPos('Brightness filter','image')/100

    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')

    s = cv2.getTrackbarPos(switch,'image')

    if s:
        todisplay = cv2.cvtColor(todisplay, cv2.COLOR_BGR2HSV)
        todisplay[...,1] = todisplay[...,1]*hue
        todisplay[...,2] = todisplay[...,2]*brightness
        todisplay = cv2.cvtColor(todisplay, cv2.COLOR_HSV2BGR)   
    else:
        todisplay[...,0] = todisplay[...,0] + b
        todisplay[...,1] = todisplay[...,1] + g
        todisplay[...,2] = todisplay[...,2] + r
 
    cv2.imshow('image',todisplay)

cv2.destroyAllWindows()