
import os

import numpy as np
from matplotlib import pyplot as plt
import cv2



def identify_phasewrap_gray(gray):
    
    # takes in a single img in hsv format
    
    # returns x,y,w,h of the rect bbox around phase-wrap ROI
    
    mask1 = cv2.inRange(gray, 10, 30)
    mask2 = cv2.inRange(gray, 30, 35)

    mask = cv2.bitwise_or(mask1, mask2)


    ret,thresh = cv2.threshold(mask, 40, 255, 0)

    im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:

        c = max(contours, key = cv2.contourArea)

        x,y,w,h = cv2.boundingRect(c)
        
        
        print(x,y,w,h) 
        
        return (x,y,w,h)

    
def draw_bbox(img,x,y,w,h):
    
    # draw rect bbox of dimensions defined by x,y,w,h in the img
    # and returns the img
    
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    
    return img


    
def load_image_and_get_roi(dir_,fname):
    
    # loads image and returns the img numpy array coordinates of the roi

    img = cv2.imread(dir_ + '/'+ fname)

    if img is None:

        raise RuntimeError

    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = img.copy()

    #rgb= cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    #gray= cv2.imread(gray_dir + '/'+ fname)

    x,y,w,h = identify_phasewrap_gray(gray)

    return (img,x,y,w,h)