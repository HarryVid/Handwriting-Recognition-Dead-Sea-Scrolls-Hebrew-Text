import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize, thin
import tqdm
import time


def segment_lines(docs_path, image_name, output_path, lines):
    # Store the save path in a variable
    save_loc = output_path + '/' + lines + \
        "/{:04d}/".format(int(image_name.split('.')[0]))
    os.makedirs(save_loc, exist_ok=True)
    # Read line image
    original_image = cv2.imread(os.path.join(
        docs_path, image_name), cv2.IMREAD_GRAYSCALE)
    line_image = original_image.copy()
    boxes_locations = []

    # Binarizing the image helps increase the segmentation accuracy
    window_size = 3
    thresh_sauvola = threshold_sauvola(line_image, window_size=window_size)
    binary_sauvola = line_image > thresh_sauvola
    binary_sauvola = np.uint8(binary_sauvola)
    # Find the contours
    ctrs, hier = cv2.findContours(
        binary_sauvola.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(tqdm.tqdm(sorted_ctrs)):
        x, y, w, h = cv2.boundingRect(ctr)

        roi = line_image[y:y + h, x:x + w]

        area = w*h
        # Check if the identified character is valid based on area size
        if area > 900 and area < 100000:
            boxes_locations.append([x, y, x + w, y + h])

    # save all identified characters
    for i, box in tqdm.tqdm(enumerate(boxes_locations)):
        x1, y1, x2, y2 = box
        character_img = original_image[y1:y2, x1:x2].copy()
        plt.imsave(save_loc + '/' +
                   "{:04d}".format(i)+'.jpg', character_img, cmap='gray')


def iterate_over_folders(docs_path, output_images_path, img_name):
    """
    This is the entry point for this module.
    args:
    docs_path: path of the folder which contains seperated lines
    output_images_path: the output directry to save the segmented files in
    img_name: name of the document
    """
    start = time.process_time()
    # Go through each line in the document folder
    for image_name in tqdm.tqdm(os.listdir(docs_path)):
        segment_lines(docs_path, image_name, output_images_path, img_name)
    print("Time taken", time.process_time() - start)
