
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage.util import invert
from heapq import *
import os

""" roate the image to predicted angle 
img --> image file
angle --> predicted angle 
"""
def rotate_angle(img,angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# computing horizontal projection on the edge detected image
def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1)

# computing the horizontal projection profile
def compute_hpp(img):
    # by edge detection
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobel_image=sobel(img_gray)
    hpp=horizontal_projections(sobel_image)
    peaks=find_peak_regions(hpp)
    peaks_index = np.array(peaks)[:,0].astype(int)

    return hpp,peaks,peaks_index

# computing the horizontal projection profile with segmented image show
def compute_hpp_with_seg(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sobel_image=sobel(img_gray)
    hpp=horizontal_projections(sobel_image)
    peaks=find_peak_regions(hpp)
    peaks_index = np.array(peaks)[:,0].astype(int)
    segmented_img = np.copy(img_gray)
    r,c = segmented_img.shape
    for ri in range(r):
        if ri in peaks_index:
            segmented_img[ri, :] = 0
    plt.figure(figsize=(10,10))
    plt.imshow(segmented_img, cmap="gray")

    plt.show()
    return hpp,peaks,peaks_index

# find peaks of horizontal projection profile
def find_peak_regions(hpp, divider=5):
    threshold = (np.max(hpp)-np.min(hpp))/divider
    peaks = []
    peaks_index = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])
    return peaks


################################ line segmentation part ##############################################
#group the peaks into walking windows
def get_hpp_walking_regions(peaks_index):
    hpp_clusters = []
    cluster = []
    for index, value in enumerate(peaks_index):
        cluster.append(value)

        if index < len(peaks_index)-1 and peaks_index[index+1] - value > 1:
            hpp_clusters.append(cluster)
            cluster = []

        #get the last cluster
        if index == len(peaks_index)-1:
            hpp_clusters.append(cluster)
            cluster = []

    return hpp_clusters

# get the binary image 
def get_binary(img):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary*1
    return binary

def path_exists(window_image):
    #very basic check first then proceed to A* check
    if 0 in horizontal_projections(window_image):
        return True

    padded_window = np.zeros((window_image.shape[0],1))
    world_map = np.hstack((padded_window, np.hstack((window_image,padded_window)) ) )
    path = np.array(astar(world_map, (int(world_map.shape[0]/2), 0), (int(world_map.shape[0]/2), world_map.shape[1])))
    if len(path) > 0:
        return True

    return False

# find the intersecting lines 
def get_road_block_regions(nmap1):
    road_blocks = []
    needtobreak = False
    for col in range(nmap1.shape[1]):
        start = col
        end = col+25
        if end > nmap1.shape[1]-1:
            end = nmap1.shape[1]-1
            needtobreak = True

        if path_exists(nmap1[:, start:end]) == False:
            road_blocks.append(col)

        if needtobreak == True:
            break

    return road_blocks

def group_the_road_blocks(road_blocks):
    #group the road blocks
    road_blocks_cluster_groups = []
    road_blocks_cluster = []
    size = len(road_blocks)
    for index, value in enumerate(road_blocks):
        road_blocks_cluster.append(value)
        if index < size-1 and (road_blocks[index+1] - road_blocks[index]) > 1:
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

        if index == size-1 and len(road_blocks_cluster) > 0:
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

    return road_blocks_cluster_groups

############################################### A* path planning #######################################################

def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal):
    #print(f"start:{start}")
    #print(f"goal:{goal}")
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
    return []


# line extraction from the input image
def extract_line_from_image(image, lower_line, upper_line):
    #print(f"upper line:{upper_line}")
    #print(f"lower line:{lower_line}")
    lower_boundary = np.min(lower_line[:, 0])
    upper_boundary = np.min(upper_line[:, 0])
    img_copy = np.copy(image)
    r, c = img_copy.shape
    for index in range(c-1):
        img_copy[0:lower_line[index, 0], index] = 255
        img_copy[upper_line[index, 0]:r, index] = 255

    return img_copy[lower_boundary:upper_boundary, :]

# seperate the lines segments into seperate images and store
def get_line_segments(img,line_segments,folder_name):
    os.chdir(folder_name)
    line_images = []
    line_count = len(line_segments)
    fig, ax = plt.subplots(figsize=(10,10), nrows=line_count-1)
    for line_index in range(line_count-1):
        #print(f"{line_index+1}:{line_segments[line_index+1]}")
        line_image = extract_line_from_image(img, line_segments[line_index], line_segments[line_index+1])
        cv2.imwrite(str(line_index)+".jpg",line_image)
        line_images.append(line_image)
        ax[line_index].imshow(line_image, cmap="gray")

    #plt.savefig("line_seg_out.jpg")
    #plt.show()

# view the single line segment (reference)
def view_single_line_seg():
    cluster_of_interest = hpp_clusters[1]
    offset_from_top = cluster_of_interest[0]
    nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
    plt.figure(figsize=(20,20))
    plt.imshow(invert(nmap), cmap="gray")

    path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
    plt.plot(path[:,1], path[:,0])
    plt.show()
