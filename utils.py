
import os

import numpy as np
#from matplotlib import pyplot as plt
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

'''
def draw_bbox(img,x,y,w,h):
    
    # draw rect bbox of dimensions defined by x,y,w,h in the img
    # and returns the img
    
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    
    return img
'''


def is_RoI_white(img):
    
    # draw rect bbox of dimensions defined by x,y,w,h in the img
    # and returns the img
    #img = cv2.imread(fname)


    #if img is None:

    #    raise RuntimeError

    #print('img shape = ',img.shape)

    #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #gray = img.copy()

    
 
    #cv2.rectangle(gray,(65,47),(120,125),(255,255,255),2)


    img = img[45:130,65:120]

    num_white_px = ((205 < img) & (img <= 255)).sum()


    num_black_px = ((0 <= img) & (img < 50)).sum()

    print("num_white_px = ", num_white_px)
    print("num_black_px = ", num_black_px)

    if (num_white_px == 0):

        return False

    elif (num_black_px == 0):

        return True

    else:

        white_to_black_ratio = num_white_px/num_black_px 

        print("white_to_black_ratio = ",white_to_black_ratio)

        if (white_to_black_ratio > 1):

            return True

        else:

            return False

    #cv2.imshow('img',gray)
    #cv2.waitKey(0)


    
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


def find_wrap(fname):
    


    img = cv2.imread(fname)


    if img is None:

        raise RuntimeError

    print('img shape = ',img.shape)

    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,5)

    RoI_is_white = is_RoI_white(img)

    gray = img.copy()
    
    
    img = img[40:130,70:130]

    

    #cv2.imshow('img',img)
    #cv2.waitKey(0)

    



    
    #170,100
    img[img > 180] = 0
    
    


    cv2.imshow('img',img)
    cv2.waitKey(0)

    ret,thresh = cv2.threshold(img, 60, 255, 0)
    thresh = cv2.bitwise_not(thresh)
    #cv2.imshow('thresh',thresh)
    #cv2.waitKey(0)
    im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:3]

    #if len(contours) != 0:

    for c in contours:
        #c = max(contours, key = cv2.contourArea)

        cnt_area = cv2.contourArea(c)
        print('max cnt area = ',cnt_area)

        if (cnt_area >= 150):

            x,y,w,h = cv2.boundingRect(c)
            x = x + 70
            y = y + 40


            print('top left coords of bbox = ',x,y)
            print('bottom left coords of bbox = ',x,(y+h))
            print('height,width = ',h,w)

            cv2.rectangle(gray,(x,y),(x+w,y+h),(255,255,255),2)

    # show the images np.hstack([hsv,rgb])
    cv2.imshow("Result", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    #dir_ = "../../books/vortex_data/vortex/vortex/4ch/intra-subject/scan_5/phase_1/x/registered-resized-1/png"
    
    dir_ = "../../books/vortex_data/vortex/vortex/4ch/inter-subject/subject_8/phase_1/y/registered-resized-1/png"
    #dir_ = "./imgs/gray_frames"
    fnames = os.listdir(dir_)

    for fname in fnames:

        impath = os.path.join(dir_,fname)
        print(impath)

        draw_bbox(impath)
        find_wrap(impath)

        

