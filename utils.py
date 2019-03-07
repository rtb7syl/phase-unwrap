
import time
import os
import random

import numpy as np
import math
#from matplotlib import pyplot as plt
import cv2

import csv

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



def get_wrapped_coords_as_list(csv_path):

        

        results = []
        with open(csv_path) as csvfile:

                reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
                for row in reader: # each row is a list
                        results.append(tuple(row))

        print(results)

        return results




def scale_wrapped_coords_wrt_roi(wrapped_coords,x,y,padding):

        #scales the wrapped coords wrt the roi region
        #and a padding of certain width all around the roi
        #x,y is the coordinate of the leftmost ,topmost point of the roi


        #x,y coord shifts due to padding
        x = x - padding
        y = y - padding

        for wrapped_coord in wrapped_coords:

                wrapped_coord[0] = wrapped_coord[0] - x
                wrapped_coord[1] = wrapped_coord[1] - y

        print(wrapped_coords)

        return wrapped_coords



def get_all_grids_with_different_strides(wrapped_coord_x,wrapped_coord_y,grid_size,cap=-1):

    initial_grid_x,initial_grid_y = get_grid_coords(wrapped_coord_x,wrapped_coord_y,grid_size)

    total_num_of_coords = grid_size**2

    coords = []

    for coord_idx in range(total_num_of_coords):

        if (coord_idx != total_num_of_coords//2):

            grid_x = coord_idx % grid_size
            grid_y = (coord_idx - grid_x)/grid_size

            coords.append((grid_x,grid_y))

    if (cap!=-1):

        #randomly sample cap num of elements from coords_population

        coords = random.sample(coords,cap)
    
    #reference coordinate which scales coordinates in coords list
    ref_x = initial_grid_x - (grid_size//2)
    ref_y = initial_grid_y - (grid_size//2)

    coords = list(map(lambda coord: ((coord[0]+ref_x),(coord[1]+ref_y)) , coords))

    return coords




def take_different_strides_and_find_candidate(wrapped_coord_x,wrapped_coord_y,grid_size,wrapped_coords):

    #proposed_grids is a list of leftmost top coords (grid_x,grid_y) of all the proposed grids with diff strides
    proposed_grids = get_all_grids_with_different_strides(wrapped_coord_x,wrapped_coord_y,grid_size)


    #this var is initially set to false,
    #but if any of the proposed grids has delta > 0,this would be set to true
    is_any_of_the_proposed_grids_a_candidate = False


    for proposed_grid in proposed_grids:

        grid_x = int(proposed_grid[0])
        grid_y = int(proposed_grid[1])
        
        strided_grid_delta = compute_delta(grid_x,grid_y,grid_size,wrapped_coords)

        if (strided_grid_delta >= 0):

            is_any_of_the_proposed_grids_a_candidate = True

            break

    return (is_any_of_the_proposed_grids_a_candidate,grid_x,grid_y)




def find_num_of_unwrapped_coords_in_grid(grid_x,grid_y,grid_size,wrapped_coords):

    num_wrapped_coords = 0

    for wrapped_coord in wrapped_coords:

        if (wrapped_coord[0] <= grid_x+grid_size  and wrapped_coord[0] >= grid_x) and (wrapped_coord[1] <= grid_y+grid_size  and wrapped_coord[1] >= grid_y):

            num_wrapped_coords = num_wrapped_coords + 1

    num_unwrapped_coords = grid_size*grid_size - num_wrapped_coords

    return num_unwrapped_coords



def get_nearest_unwrapped_coord(wrapped_coord_x,wrapped_coord_y,unwrapped_coords):

    #finds the coord in unwrapped_coords which is nearest to the wrapped_coord

    dists_from_wrapped_coord = list(map(lambda coord:(((coord[0]-wrapped_coord_x)**2)+((coord[1]-wrapped_coord_y)**2)),unwrapped_coords))

    min_dist = min(dists_from_wrapped_coord)
    
    return unwrapped_coords[dists_from_wrapped_coord.index(min_dist)]


def correct_wrapped_coord(phase_vals_matrix,wrapped_coord_x,wrapped_coord_y,grid_x,grid_y,grid_size,RoI_is_white,unwrapped_coords,wrapped_coords,fallback=False):

    #corrects a wrapped coord

    grid = phase_vals_matrix[grid_y:grid_y+grid_size,grid_x:grid_x+grid_size]

    flattened_grid = grid.flatten()

    sorted_grid_pixels = np.sort(flattened_grid)

    if (RoI_is_white):

        sorted_grid_pixels = sorted_grid_pixels[::-1]

    if (fallback == False):

        #find index of median

        mid_idx = len(sorted_grid_pixels)//2

        median = sorted_grid_pixels[mid_idx]

        compare_x,compare_y = list(zip(*np.where(grid == median)))[0]

    else:

        #fallback option,when every other heuristic fails

        num_of_unwrapped_coords = find_num_of_unwrapped_coords_in_grid(grid_x,grid_y,grid_size,wrapped_coords)

        if (num_of_unwrapped_coords > 0):

            compare_with_pixel_val = sorted_grid_pixels[num_of_unwrapped_coords - 1]

            compare_x,compare_y = list(zip(*np.where(grid == compare_with_pixel_val)))[0]

        else:

            #last option, when every other option is fucked
            
            #replace with nearest unwrapped pixel

            compare_x,compare_y = get_nearest_unwrapped_coord(wrapped_coord_x,wrapped_coord_y,unwrapped_coords)


    # now after getting the compare coordinate, we need to correct our wrapped coord based on that
    #print(compare_y,compare_x)
    compare_with_phase_val = phase_vals_matrix[int(compare_y),int(compare_x)]
    wrapped_coord_phase_val = phase_vals_matrix[wrapped_coord_y,wrapped_coord_x]

    if (RoI_is_white):

        wrapped_coord_phase_val =  wrapped_coord_phase_val + 6*math.ceil((compare_with_phase_val - wrapped_coord_phase_val)/6) 
        #wrapped_coord_phase_val = compare_with_phase_val + abs(wrapped_coord_phase_val)

    else:

        #wrapped_coord_phase_val = compare_with_phase_val  - abs(wrapped_coord_phase_val)
        wrapped_coord_phase_val =  6*math.ceil((compare_with_phase_val - wrapped_coord_phase_val)/6) -  abs(wrapped_coord_phase_val) 

    phase_vals_matrix[wrapped_coord_y,wrapped_coord_x] = wrapped_coord_phase_val

    phase_vals_matrix[wrapped_coord_y,wrapped_coord_x] = wrapped_coord_phase_val

    #phase_vals_matrix = np.uint8(phase_vals_matrix)

    #crop_matrix = phase_vals_matrix[45:130,65:120]

    
    #crop_matrix = cv2.medianBlur(crop_matrix,3)

    #phase_vals_matrix[45:130,65:120] = crop_matrix

    
    return phase_vals_matrix








    


    










def get_grid_coords(x,y,grid_size):

        #returns the leftmost,topmost coord of the grid
        #given the central coord of the grid and the grid size
        #grid has odd dimensions ie 2k+1*2k+1 grid

        scale = grid_size//2

        return (x-scale,y-scale)



def compute_delta(x,y,grid_size,all_wrapped_coords):

        #computes delta ie the difference between
        #num of unwrapped coords - num of wrapped coords
        #in a odd sq grid as provided in the args
        #x,y is the leftmost topmost pixel coordinate of the grid
        #wrt to the entire image
        #grid_size is either 3 or 5 or 7

        num_wrapped_coords = 0

        for wrapped_coord in all_wrapped_coords:

                if (wrapped_coord[0] <= x+grid_size  and wrapped_coord[0] >= x) and (wrapped_coord[1] <= y+grid_size  and wrapped_coord[1] >= y):

                        num_wrapped_coords = num_wrapped_coords + 1

        num_unwrapped_coords = grid_size*grid_size - num_wrapped_coords

        delta = num_unwrapped_coords - num_wrapped_coords

        #print(num_wrapped_coords)

        print("delta",delta)

        return delta



def compute_median(img,x,y,grid_size):

        #computes the median of the pixels in the grid
        #specified by x,y and grid_size ,in the roi
        #x,y is the leftmost topmost pixel coordinate of the grid

        x = int(x)
        y = int(y)

        median = np.median(img[y:y+grid_size, x:x+grid_size])

        print("median",median)

        return median




def replace_wrapped_pixel_with_median_pixel(img,wrapped_coord_x,wrapped_coord_y,grid_size):
        
        #replace wrapped coord with median of the grid as mentioned by the grid_size

        #find the leftmost,topmost coord of the grid
        grid_x,grid_y = get_grid_coords(wrapped_coord_x,wrapped_coord_y,grid_size)

        median = compute_median(img,grid_x,grid_y,grid_size)
        print("WC pixel val before = ",(wrapped_coord_x,wrapped_coord_y),img[wrapped_coord_x,wrapped_coord_y])

        img[wrapped_coord_y,wrapped_coord_x] = median
        print("WC pixel val after = ",img[wrapped_coord_y,wrapped_coord_x])

        return img




def replace_wrapped_pixel_with_max_unwrapped_pixel(img,delta,wrapped_coord_x,wrapped_coord_y,grid_size):

        #replaces central wrapped coord with max value of unwrapped pixel in that grid
        #subroutine is mainly called when the median pixel isn't an unwrapped pixel,
        #even in a 7*7 grid
        #in that case we replace wrapped pixel with max val of unwrapped pixel,
        #instead of the median


        
        #find the leftmost,topmost coord of the grid
        grid_x,grid_y = get_grid_coords(wrapped_coord_x,wrapped_coord_y,grid_size)

        print("grid_x,grid_y",grid_x,grid_y)
        sorted_pixels_in_grid = np.sort(img[grid_y:grid_y+grid_size, grid_x:grid_x+grid_size], axis=None)

        print(sorted_pixels_in_grid)

        #since total #pixels in grid = 49
        #and difference between wrapped and unwrapped in delta
        max_unwrapped_pixel_coord = int((49+delta)//2) - 1

        max_unwrapped_pixel = sorted_pixels_in_grid[max_unwrapped_pixel_coord]

        print(max_unwrapped_pixel_coord,max_unwrapped_pixel)

        print("WC pixel val before = ",(wrapped_coord_x,wrapped_coord_y),img[wrapped_coord_y,wrapped_coord_x])

        img[wrapped_coord_y,wrapped_coord_x] = max_unwrapped_pixel

        print("WC pixel val after = ",img[wrapped_coord_y,wrapped_coord_x])

        return img


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

    num_white_px = ((195 < img) & (img <= 255)).sum()


    num_black_px = ((0 <= img) & (img < 60)).sum()

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

def scale_pixel_coordinates(wrap_pixel_coordinates,x,y):
    
    # returns final scaled coordinates array wrt to the entire image
    
    
    scale_each_coordinate = lambda wrap_pixel_coordinate : np.array([wrap_pixel_coordinate[0] + x,wrap_pixel_coordinate[1] + y])
    wrap_pixel_coordinates = np.array(list(map(scale_each_coordinate,wrap_pixel_coordinates)))
    
    #white_pixel_coordinates = vscale_each_coordinate(white_pixel_coordinates,x,y)
    
    return wrap_pixel_coordinates



def save_pixel_coordinates_as_csv(wrap_out_fname,unwrap_out_fname,wrap_pixel_coordinates,unwrap_pixel_coordinates):
    
    # saves the numpy 2D pixel coordinates as a csv

    with open(wrap_out_fname,'a') as wf_handle:

        np.savetxt(wf_handle,wrap_pixel_coordinates,fmt='%.2e',delimiter=',')


    with open(unwrap_out_fname,'a') as uwf_handle:

        np.savetxt(uwf_handle,unwrap_pixel_coordinates,fmt='%.2e',delimiter=',')



def check_for_wrap(gray,RoI_is_white,x,y,w,h):

    #checks whether wrap is present in the region bounded by x,y,w,h

    # cropped region of image defined by x,y,w,h
    crop_img = gray[y:y+h, x:x+w]

    total_num_of_pixels = crop_img.size


    if (RoI_is_white):

        #if background is white, check for black pixels(wrapped)

        thresh = 55

        thresh_back = 200
    
        num_of_wrapped_pixels = len(crop_img[crop_img <= thresh])

        num_of_unwrapped_pixels = len(crop_img[crop_img >= thresh_back])
        
        print('num_of_wrapped_pixels = ', str(num_of_wrapped_pixels))
        print('num_of_unwrapped_pixels = ', str(num_of_unwrapped_pixels))

    
    else:
        
        #if background is black, check for white pixels(wrapped)

        thresh = 200

        thresh_back = 55
    
        num_of_wrapped_pixels = len(crop_img[crop_img >= thresh])

        num_of_unwrapped_pixels = len(crop_img[crop_img <= thresh_back])
        
        print('num_of_wrapped_pixels = ', str(num_of_wrapped_pixels))
        print('num_of_unwrapped_pixels = ', str(num_of_unwrapped_pixels))
    
    frac_of_wrapped_pixels = num_of_wrapped_pixels/total_num_of_pixels
    frac_of_unwrapped_pixels = num_of_unwrapped_pixels/total_num_of_pixels

    print('frac_of_wrapped_pixels',frac_of_wrapped_pixels)
    print('frac_of_unwrapped_pixels',frac_of_unwrapped_pixels)
    
    if (frac_of_unwrapped_pixels >= 0.1 and num_of_wrapped_pixels > 0):

        #if,there's wrap in the region return True

        return (True,thresh,num_of_wrapped_pixels)

    else:
        
        return (False,thresh,num_of_wrapped_pixels)

    
def grab_wrapped_and_unwrapped_coords_and_save_to_csv(gray,RoI_is_white,x,y,w,h,wrap_out_fname,unwrap_out_fname):

    #grabs the wrapped coords from the region bounded by x,y,w,h and saves them in a csv file

    crop_img = gray[y:y+h, x:x+w]

    if (RoI_is_white):

        thresh = 105
        wrap_indices = np.nonzero(crop_img <= thresh)
        unwrap_indices = np.nonzero(crop_img > thresh)

    else:
        thresh = 150
        wrap_indices = np.nonzero(crop_img >= thresh)
        unwrap_indices = np.nonzero(crop_img < thresh)

    #print(indices)
    wrap_y_indices = wrap_indices[0]
    wrap_x_indices = wrap_indices[1]

    unwrap_y_indices = unwrap_indices[0]
    unwrap_x_indices = unwrap_indices[1]
    
    # array of wrap px coordinates with respect to the cropped sq.
    wrap_pixel_coordinates = np.column_stack((wrap_x_indices,wrap_y_indices)) 

    wrap_pixel_coordinates = scale_pixel_coordinates(wrap_pixel_coordinates,x,y)

    # array of unwrap px coordinates with respect to the cropped sq.
    unwrap_pixel_coordinates = np.column_stack((unwrap_x_indices,unwrap_y_indices)) 

    unwrap_pixel_coordinates = scale_pixel_coordinates(unwrap_pixel_coordinates,x,y)
    #print('wrap_pixel_coordinates\n')

    #print(wrap_pixel_coordinates)

    save_pixel_coordinates_as_csv(wrap_out_fname,unwrap_out_fname,wrap_pixel_coordinates,unwrap_pixel_coordinates)





def find_wrap(fname,wrap_out_fname,unwrap_out_fname,contour_boxes_fname):

    


    img = cv2.imread(fname)


    if img is None:

        raise RuntimeError

    print('img shape = ',img.shape)

    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    RoI_is_white = is_RoI_white(img)
    print('RoI_is_white',RoI_is_white)

    

    

    gray = img.copy()

    
    
    #cropping the img to find contours in this region
    img = img[40:130,70:130]

    img = cv2.medianBlur(img,5)




    

    #cv2.imshow('img',img)
    #cv2.waitKey(0)

    



    
    #170,100
    img[img > 180] = 0
    
    


    #cv2.imshow('img',img)
    #cv2.waitKey(0)

    ret,thresh = cv2.threshold(img, 60, 255, 0)
    thresh = cv2.bitwise_not(thresh)

    #cv2.imshow('thresh',thresh)
    #cv2.waitKey(0)
    

    im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:3]


    wrapped_cnts = []

    #if len(contours) != 0:
    
    total_num_of_wrapped_pixels = 0


    for c in contours:
        #c = max(contours, key = cv2.contourArea)

        cnt_area = cv2.contourArea(c)

        print('cnt area = ',cnt_area)

        x,y,w,h = cv2.boundingRect(c)

        print('bbox_area = ',str(w*h))

        if (cnt_area >= 120):

            #x,y,w,h = cv2.boundingRect(c)

            x = x - 2
            y = y - 2
            w = w + 1
            h = h + 1



            print('bbox_area = ',str(w*h))

            #scaling the coordinates of the crop img to fit in the original img

            x = x + 70
            y = y + 40


            print('top left coords of bbox = ',x,y)
            print('bottom left coords of bbox = ',x,(y+h))
            print('height,width = ',h,w)

            #cv2.rectangle(gray,(65,47),(120,125),(255,255,255),2)
            

            #check for wrap in contour
            is_wrap,wrap_px_thresh,num_of_wrapped_pixels = check_for_wrap(gray,RoI_is_white,x,y,w,h)
            print('wrap_px_thresh',wrap_px_thresh)
            
            if (is_wrap):

                #if this is a wrapped contour, 
                #grab the wrapped coords and save them in a csv file
                #print('num_of_wrapped_pixels',num_of_wrapped_pixels)

                total_num_of_wrapped_pixels = total_num_of_wrapped_pixels + num_of_wrapped_pixels

                grab_wrapped_and_unwrapped_coords_and_save_to_csv(gray,RoI_is_white,x,y,w,h,wrap_out_fname,unwrap_out_fname)

                wrapped_cnts.append([x,y,w,h])

            #cv2.rectangle(gray,(x,y),(x+w,y+h),(255,255,255),2)

    #print(wrapped_cnts)
    
    print('total_num_of_wrapped_pixels',total_num_of_wrapped_pixels)
    
    if (len(wrapped_cnts) != 0):
        
        #if there are wrapped regions in the frame, 
        #save the box coords in a csv file
        '''
        for cnt in wrapped_cnts:

            x = cnt[0]
            y = cnt[1]
            w = cnt[2]
            h = cnt[3]

            cv2.rectangle(gray,(x,y),(x+w,y+h),(255,255,255),1)
        '''

        if os.path.exists(contour_boxes_fname) == False:
            with open(contour_boxes_fname,'w') as wf_handle:

                np.savetxt(wf_handle,np.array(wrapped_cnts),fmt='%.2e',delimiter=',')



    
    if (total_num_of_wrapped_pixels > 0):

        return gray
    
    else:
        
        return None
    

    '''

    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,255,255),2)


    # show the images np.hstack([hsv,rgb])
    cv2.imshow("Result", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''


        











    '''
    #wrapped_cnts_copy = wrapped_cnts.copy()


    
    #subroutine for unwrapping, if there are any wrapped RoIs in the frame
    if (len(wrapped_cnts) != 0):

        for i in range(len(wrapped_cnts)):
                wrapped_cnt = wrapped_cnts[i]

                del wrapped_cnts_copy[0]

                #replace other wrapped cnts with mean of unwrapped coords in those cnts
                mean_replaced_img = gray.copy()
                
                for j in range(len(wrapped_cnts_copy)):
                

                wrapped_cnt_to_be_replaced = wrapped_cnts_copy[j]

                mean_replaced_img = replace_wrapped_pixels_with_mean_of_unwrapped_pixels(mean_replaced_img,
                                  dir_ = dir_prefix + meta_data_dict["type"] + '/' + entity + '_'                                                              
                                                                                                wrapped_cnt_to_be_replaced,
                                                                                                
                                                                                                RoI_is_white
                                                                                                
                                                                                                )

                gray = unwrap(mean_replaced_img,gray,wrapped_cnt,RoI_is_white)

    '''







    















    








if __name__ == "__main__":
    '''
    #dir_ = "../../books/vortex_data/vortex/vortex/4ch/intra-subject/scan_5/phase_1/x/registered-resized-1/png"
    
    dir_ = "../../books/vortex_data/vortex/vortex/4ch/inter-subject/subject_8/phase_1/y/registered-resized-1/png"
    #dir_ = "./imgs/gray_frames"
    fnames = os.listdir(dir_)

    for fname in fnames:

        impath = os.path.join(dir_,fname)
        print(impath)

        draw_bbox(impath)
        find_wrap(impath)

    '''


