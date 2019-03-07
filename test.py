from unwrap import *
from utils import *
from matplotlib import pyplot as plt
import time
from utils import find_wrap
def generate_impaths_of_all_images(dir_prefix):

    #subroutine generator to generate impaths and txtpaths of every image
    #and corresponding txt files in the dataset on the fly
    

    meta_data_dicts = [{"type":"intra-subject","scan":["1","2","3","4","5"],"phase":["1","2"],"direction":["x","y"]},
    
                      {"type":"inter-subject","subject":["1","2","3","4","5","6","7","8"],"phase":["1"],"direction":["x","y"]}]

    for meta_data_dict in meta_data_dicts:

        #dir_ = ''

        if (meta_data_dict["type"] == "intra-subject"):

            entity = "scan"

        else:

            entity = "subject"

        dir_seg_1 = dir_prefix + meta_data_dict["type"] + '/' + entity + '_'

        for entity_i in meta_data_dict[entity]:

            dir_seg_2 = dir_seg_1 + entity_i + '/'

            for phase_i in meta_data_dict["phase"]:

                dir_seg_3 = dir_seg_2 + "phase_" + phase_i + '/'

                for direction_i in meta_data_dict["direction"]:

                
                    #the final directories

                    txt_dir = dir_seg_3 + direction_i + '/registered-resized-1' 
                    im_dir = txt_dir + '/png'
                    #print(dir_)

                    im_fnames = os.listdir(im_dir)

                    for im_fname in im_fnames:

                        if (im_fname != "animation"):
                            
                            
                            impath = os.path.join(im_dir,im_fname)

                            txt_fname = im_fname.split('.')[0]+'.txt'
                            txt_path = os.path.join(txt_dir,txt_fname)
                            
                            print(impath,txt_path)

                            yield (impath,txt_path)



def detect_wrap_and_save_coords_as_csv(impath,out_csv_prefix,out_imgs_prefix):


    #given the impath of an img,detects wrap,and
    #writes the wrapped coords and the wrap detected frame to corresponding dirs
    
    impath_splitted = impath.split('/')
    imname = impath_splitted[-1].split('.')[0]

    fname_suffix = '/'.join(impath_splitted[9:13])+'/'+imname



    #print(fname_suffix)

    wrap_out_csv_fname = out_csv_prefix + '/' + fname_suffix + '_wrap.csv'
    unwrap_out_csv_fname = out_csv_prefix + '/' + fname_suffix + '_unwrap.csv'
    contour_boxes_fname = out_csv_prefix + '/' + fname_suffix + '_bboxes.csv'
    #out_imgs_fname = out_imgs_prefix + '/' + fname_suffix + '.png'

    print(wrap_out_csv_fname,unwrap_out_csv_fname)

    wrap_detected_img = find_wrap(impath,wrap_out_csv_fname,unwrap_out_csv_fname,contour_boxes_fname)

    #if (wrap_detected_img is not None):

    #    cv2.imwrite(out_imgs_fname, wrap_detected_img)







def test_unwrap_algorithm(img,text_path,wrap_csv_path,unwrap_csv_path,contour_boxes_csv_path,out_imgs_fname,out_txts_fname):

    #reads img,corresponding text and csv data from corresponding paths 
    #and passes the data to the unwrap pipeline
    '''
    img = cv2.imread(impath)

    

    if img is None:

        raise RuntimeError

    print('img shape = ',img.shape)

    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #original_img = img.copy()
    '''


    out = '.'+out_imgs_fname.split('.')[1]+'_original.'+out_imgs_fname.split('.')[2]
    print('out1',out)
    cv2.imwrite(out,img)

    
    
    #cv2.rectangle(original_img,(60,45),(120,130),(255,255,255),2)

    RoI_is_white = is_RoI_white(img)
    print('RoI_is_white',RoI_is_white)

    wrapped_coords = get_wrapped_coords_as_list(wrap_csv_path)
    unwrapped_coords = get_wrapped_coords_as_list(unwrap_csv_path)
    contour_boxes = get_wrapped_coords_as_list(contour_boxes_csv_path)

    img = np.genfromtxt(text_path,np.float32)

    img = cv2.medianBlur(img,3)


    img = AdaBlur(img,RoI_is_white,wrapped_coords,unwrapped_coords)

    #plt.imshow(phase_vals_matrix,cmap='gray')

    #out = '.'+out_imgs_fname.split('.')[1]+'_after_adablur.'+out_imgs_fname.split('.')[2]
    #print('out2',out)
    #plt.imsave(out,img,cmap='gray')

    #time.sleep(5)

    #img = cv2.imread(out)
    


    for box in contour_boxes:

        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])


        crop_img = img[y-3:y+h+3,x-3:x+w+3]

        
        crop_img = cv2.medianBlur(crop_img,5)

        img[y-3:y+h+3,x-3:x+w+3] = crop_img

    np.savetxt(out_txts_fname, img, fmt='%1.10f')
    print('unwrapped phase txt written')

    out = '.'+out_imgs_fname.split('.')[1]+'_final.'+out_imgs_fname.split('.')[2]
    print('out2',out)


    plt.imsave(out,img,cmap='gray')

    '''
    cv2.imshow('original image',original_img)
    #cv2.imshow('after adablur',after_adablur_img)
    #cv2.imshow('final',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    



def test_unwrap_algorithm_driver(impath,txt_path,out_csv_prefix,out_imgs_prefix,out_txts_prefix):

    #given the impath of an img,generates paths of the
    #image,txt and csv paths and passes them to the test_unwrap method

    
    impath_splitted = impath.split('/')
    imname = impath_splitted[-1].split('.')[0]

    fname_suffix = '/'.join(impath_splitted[9:13])+'/'+imname



    print(fname_suffix)

    wrap_out_csv_fname = out_csv_prefix + '/' + fname_suffix + '_wrap.csv'
    unwrap_out_csv_fname = out_csv_prefix + '/' + fname_suffix + '_unwrap.csv'
    contour_boxes_out_csv_fname = out_csv_prefix + '/' + fname_suffix + '_bboxes.csv'
    out_imgs_fname = out_imgs_prefix + '/' + fname_suffix + '.png'
    out_txts_fname = out_txts_prefix + '/' + fname_suffix + '.txt'

    print(wrap_out_csv_fname,unwrap_out_csv_fname,contour_boxes_out_csv_fname,out_imgs_fname,out_txts_fname)

    wrap_detected_img = find_wrap(impath,wrap_out_csv_fname,unwrap_out_csv_fname,contour_boxes_out_csv_fname)
    print('.......Coords and Bboxes written............')
    
    

    if (wrap_detected_img is not None):

        time.sleep(1)

        test_unwrap_algorithm(wrap_detected_img,txt_path,wrap_out_csv_fname,unwrap_out_csv_fname,contour_boxes_out_csv_fname,out_imgs_fname,out_txts_fname)

    













if __name__ == "__main__":

    #dir_ = "/media/rtb7syl/New Volume/books/vortex_data/vortex/vortex/4ch/intra-subject/scan_1/phase_1/x/registered-resized-1/png"
    '''
    dir_ = "/media/rtb7syl/New Volume/books/vortex_data/vortex/vortex/4ch/inter-subject/subject_5/phase_1/y/registered-resized-1/png"
    impath = os.path.join(dir_,'32.png')
    
    

    wrap_csv_path = "./results/wrapped_coords/inter-subject/subject_5/phase_1/y/32_wrap.csv"
    unwrap_csv_path = "./results/wrapped_coords/inter-subject/subject_5/phase_1/y/32_unwrap.csv"

    contour_boxes_csv_path = "./results/wrapped_coords/inter-subject/subject_5/phase_1/y/32_bboxes.csv"

    gray = find_wrap(impath,wrap_csv_path,unwrap_csv_path,contour_boxes_csv_path)

    text_path = "/media/rtb7syl/New Volume/books/vortex_data/vortex/vortex/4ch/inter-subject/subject_5/phase_1/y/registered-resized-1/32.txt"
    out_imgs_fname = ''
    test_unwrap_algorithm(impath,text_path,wrap_csv_path,unwrap_csv_path,contour_boxes_csv_path,out_imgs_fname)

    '''

    #find_wrap(impath,'x.csv')

    #dir_ = "./imgs/gray_frames"
    
    #print(os.listdir(dir_))
    '''
    #dir_ = "./imgs/gray_frames"
    fnames = os.listdir(dir_)

    for fname in fnames:

        impath = os.path.join(dir_,fname)
        print(impath)

        #draw_bbox(impath)
        #find_wrap(impath)
    '''
    '''
    impath = os.path.join(dir_,'37.jpg')
    print(impath)

    out_fname = '37.csv'
    find_wrap(impath,out_fname)
    '''
    #detect_wrap_and_save_coords_as_csv("/media/rtb7syl/New Volume/books/vortex_data/vortex/vortex/4ch/intra-subject/scan_5/phase_1/x/registered-resized-1/png/30.png","./results/wrapped_coords","./results/wrapped_detected_frames")
    
    
    start =  time.time()

    dir_prefix = "/media/rtb7syl/New Volume/books/vortex_data/vortex/vortex/4ch/"

    out_csv_prefix = "./wrap_coords_v2"
    out_imgs_prefix = "./wrap_corrected_imgs"
    out_txts_prefix = "./wrap_corrected_txts"

    

    for (impath,txt_path) in generate_impaths_of_all_images(dir_prefix):


        print("Processing img : ",impath)
        print("Processing txt : ",txt_path)



        test_unwrap_algorithm_driver(impath,txt_path,out_csv_prefix,out_imgs_prefix,out_txts_prefix)

        print("unwrapped frames written to dir")
    
    
    end =  time.time()

    exec_time = end - start

    print("sub-routine finished in ",exec_time," ms")

    