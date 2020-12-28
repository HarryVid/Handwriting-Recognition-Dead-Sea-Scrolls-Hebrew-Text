# Line segmentation part ------------------

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage.util import invert
from heapq import *
from .utils import *
import os
import time
import argparse
import shutil
import tqdm


""" main run of the line segmentation part
img_name --> input image
folder_name --> name of the output folder
current directory --> path of the main python file
"""
def  run(img_name,folder_name,curr_dir):
    # preprossing  -----------------------------------
    if folder_name in os.listdir():
        try:
            shutil.rmtree(folder_name)
        except OSError as e:
            print(f"Error:{e}")
            
    os.chdir(curr_dir)

    if not folder_name.split('/')[-1] in os.listdir(folder_name.split('/')[0]):
        os.mkdir(folder_name)
    else:
        shutil.rmtree(folder_name)
        os.mkdir(folder_name)

    # variables
    start=time.time()
    max_hpp_lst=[]
    angle_lst=[]
    peak_index_lst=[]

    # read image and show
    img=cv2.imread(img_name)
    #width=720 #width=img.shape[0]/4
    #height=630 #height=img.shape[1]/4
    #img=cv2.resize(img,(int(width),int(height)))
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray=cv2.bitwise_not(img_gray)
    thresh=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # compute rotation and find best suitable angle
    for angle in tqdm.tqdm(np.arange(-10,10,0.1)):
        rotated=rotate_angle(img,angle)
        # compute hpp
        hpp,peak,peak_index=compute_hpp(rotated)
        max_hpp=np.max(hpp)
        max_hpp_lst.append(max_hpp)
        angle_lst.append(angle)

    max_peak=np.max(max_hpp_lst)
    max_peak_index=max_hpp_lst.index(max_peak)
    angle_selected=angle_lst[max_peak_index]

    print(f"Max hpp list:{max_peak}")
    print(f"Max hpp index:{max_peak_index}")
    print(f"Appropriate angle :{ angle_selected}")

    # Perform line segmentation for the selected angle
    new_rotated=rotate_angle(img,angle_selected)

    # compute horizontal projection profile on the selected image
    final_hpp,final_peak,final_peak_index=compute_hpp(new_rotated)
    #print(f"final peak index:{final_peak_index}")

    # plotting image for comparison
    """new_rotated_copy=new_rotated
    cv2.putText(new_rotated_copy, "Angle: {:.2f} degrees".format(angle_selected),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    new_image=np.concatenate((img,new_rotated_copy),axis=1)
    cv2.imwrite("angle_saved.jpg",new_image)
    #cv2.imshow(f"image:{angle_selected}",new_image)
    """

    #################################################################################
    # line segmentation part

    # group the best matched images into clusters
    hpp_clusters = get_hpp_walking_regions(final_peak_index)
    binary_image = get_binary(cv2.cvtColor(new_rotated,cv2.COLOR_BGR2GRAY))
    # remove cluster with one value
    hpp_clusters=[cluster for cluster in hpp_clusters if len(cluster)!=1]
    for cluster_of_interest in tqdm.tqdm(hpp_clusters):
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
        # find regions with intersections
        road_blocks = get_road_block_regions(nmap)
        road_blocks_cluster_groups = group_the_road_blocks(road_blocks)
        #create the doorways
        for index, road_blocks in tqdm.tqdm(enumerate(road_blocks_cluster_groups)):
            window_image = nmap[:, road_blocks[0]: road_blocks[1]+10]
            binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:][:, road_blocks[0]: road_blocks[1]+10][int(window_image.shape[0]/2),:] *= 0

    #now that everything is cleaner, its time to segment all the lines using the A* algorithm
    line_segments = []

    for i, cluster_of_interest in tqdm.tqdm(enumerate(hpp_clusters)):
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
        path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
        offset_from_top = cluster_of_interest[0]
        if not len(path)==0:
            path[:,0] += offset_from_top
            line_segments.append(path)

    offset_from_top = cluster_of_interest[0]
    fig, ax = plt.subplots(figsize=(20,10), ncols=2)
    for path in line_segments:
        if len(path) is not 0:
            ax[1].plot((path[:,1]), path[:,0])
    ax[1].axis("off")
    ax[0].axis("off")
    ax[1].imshow(img, cmap="gray")
    ax[0].imshow(img, cmap="gray")
    #plt.savefig(folder_name+"/line_seg_path.jpg")
    #plt.show()
    last_bottom_row = np.flip(np.column_stack(((np.ones((new_rotated.shape[1],))*new_rotated.shape[0]), np.arange(new_rotated.shape[1]))).astype(int), axis=0)
    line_segments.append(last_bottom_row)
    
    # Generate the segmented images from the line segments
    get_line_segments(cv2.cvtColor(new_rotated,cv2.COLOR_BGR2GRAY),line_segments,folder_name)
    end=time.time()
    print(f"total time for line segmentation image {img_name} : {end-start}")
