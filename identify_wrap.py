import os

import numpy as np
from matplotlib import pyplot as plt
import cv2

from utils import *






def capture_white_pixels(img,white_thresh,x,y,w,h):
    
    # captures the white pixels in the cropped region
    # and returns them with relative to the cropped region
    # white_thresh is the min white pixel intensity 
    
    
    
    # cropped region of image defined by x,y,w,h
    crop_img = img[y:y+h, x:x+w]
    
    num_of_white_pixels = len(crop_img[crop_img >= white_thresh])
    
    print('num_of_white_pixels = ', str(num_of_white_pixels))
    
    print('white pixel coordinates\n')
    
    indices = np.nonzero(crop_img >= white_thresh)
    
    print(indices)
    y_indices = indices[0]
    x_indices = indices[1]
    
    # array of white px coordinates with respect to the cropped sq.
    white_pixel_coordinates = np.column_stack((x_indices,y_indices)) 
    
    print(white_pixel_coordinates)
    
    return (white_pixel_coordinates,num_of_white_pixels)


'''
def scale_each_coordinate(white_pixel_coordinate,x,y):
    
    #scaled_coordinate = np.array([white_pixel_coordinate[0] + x,white_pixel_coordinate[1] + y]) 
    print(white_pixel_coordinate)
    return white_pixel_coordinate
'''    
    
def scale_pixel_coordinates(white_pixel_coordinates,x,y):
    
    # returns final scaled coordinates array wrt to the entire image
    
    
    scale_each_coordinate = lambda white_pixel_coordinate : np.array([white_pixel_coordinate[0] + x,white_pixel_coordinate[1] + y])
    white_pixel_coordinates = np.array(list(map(scale_each_coordinate,white_pixel_coordinates)))
    
    #white_pixel_coordinates = vscale_each_coordinate(white_pixel_coordinates,x,y)
    
    return white_pixel_coordinates



def save_white_pixel_coordinates_as_csv(csv_fname,white_pixel_coordinates):
    
    # saves the numpy 2D pixel coordinates as a csv 
    
    np.savetxt(csv_fname,white_pixel_coordinates,fmt='%.2e',delimiter=',')
    
    
    

    


    


def check_img_for_wrap_and_write_coords_to_csv(dir_,img_fname,out_fname,sq_area_thresh,white_thresh):
    
    '''
    dir_ = './gray_frames'
    fname = '41.jpg'
    sq_area_thresh = 400
    '''
    
    img,x,y,w,h = load_image_and_get_roi(dir_,img_fname)
    
    if (w * h) >= sq_area_thresh:
        
        white_pixel_coordinates,num_of_white_pixels = capture_white_pixels(img,white_thresh,x,y,w,h)
        
        if (num_of_white_pixels > 0):

            white_pixel_coordinates = scale_pixel_coordinates(white_pixel_coordinates,x,y)
            
            print('white pixel coordinates\n')

            print(white_pixel_coordinates)

            save_white_pixel_coordinates_as_csv(out_fname,white_pixel_coordinates)
