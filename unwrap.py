import os


import numpy as np
import cv2

from utils import *




def correct_phase_wrap(img,fname,x,y,w,h):

    # cropped region of image defined by x,y,w,h
    
    gray = img.copy()
    
    crop_img = img[y:y+h, x:x+w]
    
    out_img = cv2.medianBlur(crop_img, 7)
    
    img[y:y+h, x:x+w] = out_img
    
    #imgs = np.hstack([gray,img])
        
    cv2.imwrite('./wrap_corrected/'+'corrected_'+fname,img)
    print('done')
    
    #cv2.imshow('v',imgs)
    
    #cv2.imshow('vv',crop_img)
    #cv2.imshow('vv1',out_img)
    #cv2.waitKey(0)
    
    
def main():
    
    dir_ = './gray_frames'
    
    fname = '40.jpg'
    
    '''
    for fname in os.listdir(dir_):
        
        out_fname = fname.split('.')[0] + '.csv'
        
        sq_area_thresh = 400
        
        white_thresh = 210
        
        check_img_for_wrap_and_write_coords_to_csv(dir_,fname,out_fname,sq_area_thresh,white_thresh)
        
    '''
    
    img,x,y,w,h = load_image_and_get_roi(dir_,fname)
    correct_phase_wrap(img,fname,x,y,w,h)
     
        
        
        
        
#load_image_and_get_roi('./gray_frames','38.jpg')       
main()