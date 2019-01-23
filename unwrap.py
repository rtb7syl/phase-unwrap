import os


import numpy as np
import cv2

from utils import *




def correct_phase_wrap_median_blur(img,x,y,w,h):

    # cropped region of image defined by x,y,w,h
    
    #gray1 = img.copy()
    
    crop_img = img[y:y+h, x:x+w]
    
    out_img = cv2.medianBlur(crop_img, 7)
    
    img[y:y+h, x:x+w] = out_img
    
    #imgs = np.hstack([gray,gray1,img])
        
    #cv2.imwrite('./wrap_corrected_AdaBlur+MedianBlur/'+'corrected_'+fname,img)
    #print('done')
    
    #cv2.imshow('v',imgs)
    
    #cv2.imshow('vv',crop_img)
    #cv2.imshow('vv1',out_img)
    #cv2.waitKey(0)

    return img
    



def AdaBlur(img,wrapped_coords):



    # mapping from each wrapped coord to it's corressponding delta
    #first we compute delta on a 3*3 grid with the wrapped coord at the centre
    #but if needed we can increase the size of the grid to 5*5 or 7*7
    #obviously then ,delta has to be updated
    delta_dict = {}



    for wrapped_coord in wrapped_coords:

        #take a 3*3 window/grid keeping the wrapped coord at the centre of the grid
        grid_size = 3

        wrapped_coord_x = wrapped_coord[0]
        wrapped_coord_y = wrapped_coord[1]


        #find the leftmost,topmost coord of the grid
        grid_x,grid_y = get_grid_coords(wrapped_coord_x,wrapped_coord_y,grid_size)

        print("grid_x,grid_y",grid_x,grid_y)

        #delta = #unwrapped coords - #wrapped coords, in the grid
        delta = compute_delta(grid_x,grid_y,grid_size,wrapped_coords)

        delta_dict[(wrapped_coord_x,wrapped_coord_y)] = delta

    
    #print(delta_dict)

    sorted_by_delta_list = sorted(delta_dict.items(), key=lambda kv: kv[1])

    sorted_by_delta_list.reverse()

    print(sorted_by_delta_list)


    for wrapped_coord_delta in sorted_by_delta_list:

        wrapped_coord = wrapped_coord_delta[0]

        wrapped_coord_x = int(wrapped_coord[0])
        wrapped_coord_y = int(wrapped_coord[1])

        delta = wrapped_coord_delta[1]


        if (delta >= 0):

            #take a 3*3 window/grid keeping the wrapped coord at the centre of the grid
            grid_size = 3

            #wrapped_coord_x = wrapped_coord[0]
            #wrapped_coord_y = wrapped_coord[1]

            img = replace_wrapped_pixel_with_median_pixel(img,wrapped_coord_x,wrapped_coord_y,grid_size)



    for wrapped_coord_delta in sorted_by_delta_list:

        wrapped_coord = wrapped_coord_delta[0]

        wrapped_coord_x = int(wrapped_coord[0])
        wrapped_coord_y = int(wrapped_coord[1])

        delta = wrapped_coord_delta[1]

        if (delta < 0):

            grid_size = 5

            #find the leftmost,topmost coord of the grid
            grid_x,grid_y = get_grid_coords(wrapped_coord_x,wrapped_coord_y,grid_size)

            print("grid_x,grid_y",grid_x,grid_y)

            #delta = #unwrapped coords - #wrapped coords, in the grid
            delta = compute_delta(grid_x,grid_y,grid_size,wrapped_coords)
            print("grid 5 delta",delta)

            if (delta >= 0):

                img = replace_wrapped_pixel_with_median_pixel(img,wrapped_coord_x,wrapped_coord_y,grid_size)

            elif (delta < 0):

                grid_size = 7


                #img = replace_wrapped_pixel_with_median_pixel(img,wrapped_coord_x,wrapped_coord_y,grid_size)

                delta = compute_delta(grid_x,grid_y,grid_size,wrapped_coords)
                print("grid 7 delta",delta)

                if (delta >= 0):

                    img = replace_wrapped_pixel_with_median_pixel(img,wrapped_coord_x,wrapped_coord_y,grid_size)

                elif (delta < 0):

                    #replace wrapped coord with max unwrapped pixel value

                    img = replace_wrapped_pixel_with_max_unwrapped_pixel(img,delta,wrapped_coord_x,wrapped_coord_y,grid_size)



    
    return img







            










def correct_phase_wrap_ada_blur(img,csv_path,padding,x,y,w,h):

    # cropped region of image defined by x,y,w,h
    
    #gray = img.copy()
    


    wrapped_coords = get_wrapped_coords_as_list(csv_path)

    #scale wrapped coords wrt to the roi with padding
    wrapped_coords = scale_wrapped_coords_wrt_roi(wrapped_coords,x,y,padding)
    print("wrapped_coords",wrapped_coords)

    #crop_img = img[y-padding:y+h+padding, x-padding:x+w+padding]
    crop_img = img[y:y+h, x:x+w]

    padded_img = np.pad(crop_img, ((3, 3), (3, 3)), 'constant',constant_values=0)


    
    padded_img_copy = padded_img.copy()


    #crop_img = AdaBlur(crop_img,wrapped_coords)
    
    out_img = AdaBlur(padded_img,wrapped_coords)
    
    #img[y-padding:y+h+padding, x-padding:x+w+padding] = out_img
    img[y:y+h, x:x+w] = out_img[3:-3,3:-3]

    return img
    
    #imgs = np.hstack([gray,img])
        
    #cv2.imwrite('./wrap_corrected/'+'corrected_'+fname,img)
    
    #cv2.imshow('out',imgs)
    #cv2.imshow('in',crop_img)
    #cv2.waitKey(0)
    #print('done')


def main():
    

    
    '''
    for fname in os.listdir(dir_):
        
        out_fname = fname.split('.')[0] + '.csv'
        
        sq_area_thresh = 400
        
        white_thresh = 210
        
        check_img_for_wrap_and_write_coords_to_csv(dir_,fname,out_fname,sq_area_thresh,white_thresh)
        
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX

    dir_ = './imgs/gray_frames'
    csv_dir = './data/wrapped_coordinates'
    
    fname = '40.jpg'
    csv_path = './data/wrapped_coordinates/40.csv'
    
    img,x,y,w,h = load_image_and_get_roi(dir_,fname)
    
    img_copy1 = img.copy()
    img_copy2 = img.copy()

    cv2.putText(img_copy2,'Original',(20,160), font, .8,(255,0,0),1,cv2.LINE_AA)

    #AdaBlur
    ada = correct_phase_wrap_ada_blur(img,csv_path,3,x,y,w,h)
    ada_copy = ada.copy()

    cv2.putText(ada_copy,'AdaBlur',(20,160), font, .8,(255,0,0),1,cv2.LINE_AA)
    
    #AdaBlur+MedianBlur
    ada_med = correct_phase_wrap_median_blur(ada,x,y,w,h)

    cv2.putText(ada_med,'AdaBlur+MedianBlur',(1,160), font, .6,(255,0,0),1,cv2.LINE_AA)

    #MedianBlur
    med = correct_phase_wrap_median_blur(img_copy1,x,y,w,h)

    cv2.putText(med,'MedianBlur',(17,160), font, .8,(255,0,0),1,cv2.LINE_AA)


    imgs = np.hstack([img_copy2,med,ada_copy,ada_med])
        
    cv2.imwrite('./results/'+'corrected_'+fname,imgs)
    
    #cv2.imshow('out',imgs)
    #cv2.imshow('in',crop_img)
    #cv2.waitKey(0)
    print('done')


     
        
        
        
        
#load_image_and_get_roi('./gray_frames','38.jpg')       
main()