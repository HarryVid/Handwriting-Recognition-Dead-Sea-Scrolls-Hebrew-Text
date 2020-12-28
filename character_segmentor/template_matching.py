
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import tqdm
import os
import operator
import random
import argparse
import time


output_save_path = ''
input_path = ''
character_image_path = ""
categories = []

unique_counter = 1

method = eval('cv2.TM_CCOEFF_NORMED')

orig_x = None
orig_y = None


def get_match_scores(search_image, threshold=0.52):
    match_scores = {}
    for label in categories:
        match_scores[label] = []
    for label in categories:
        # Do template matching on all the images of all the categories.
        for path in glob.glob(character_image_path+label+'/*.pgm'):
            template = cv2.imread(path, 0)
            template_x, template_y = template.shape
            scale_ratio = orig_x/template_x
            if scale_ratio > 2:  # Experiments found this scale better
                scale_ratio = 2
            search_image_x, search_image_y = search_image.shape

            y_ratio = (int)(scale_ratio*template_x)
            if y_ratio > search_image_x:
                y_ratio = search_image_x
            x_ratio = (int)(scale_ratio*template_y)
            if x_ratio > search_image_y:
                x_ratio = search_image_y
            if x_ratio <= 0 or y_ratio <= 0:
                continue
            template = cv2.resize(
                template, (x_ratio, y_ratio), interpolation=cv2.INTER_AREA)

            w, h = template.shape[::-1]
            # Incase the resize dimentions are not valid. Happens if the split image has one of the dimension as 0
            try:
                res = cv2.matchTemplate(search_image, template, method)
            except Exception as e:
                print(e)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            match_scores[label].append(max_val)
    if len(match_scores) == 0:
        return
    for label in categories:
        try:
            match_scores[label] = max(match_scores[label])
        except Exception as e:
            print(e)
            print(label)
            print(match_scores)
            return
        if match_scores[label] < threshold:
            match_scores.pop(label, None)
    return match_scores


def split_images(input_image, top_left, h, w):
    """Split the image into three regions.
    first_image is assumed to be before the extracted region.
    extracted_image is the center image
    second_image is the third image
    returns:
    three tuples of extracted image and its area ratio
    """
    extracted_image = input_image.copy(
    )[:top_left[1]+h, top_left[0]: top_left[0]+w]
    first_image = input_image.copy()[:top_left[1]+h, :top_left[0]]
    second_image = input_image.copy()[:top_left[1]+h, top_left[0]+w:]

    first_image_area_ratio = (
        first_image.shape[0]*first_image.shape[1])/(input_image.shape[0]*input_image.shape[1])

    extracted_image_area_ratio = (
        extracted_image.shape[0]*extracted_image.shape[1])/(input_image.shape[0]*input_image.shape[1])

    second_image_area_ratio = (
        second_image.shape[0]*second_image.shape[1])/(input_image.shape[0]*input_image.shape[1])

    return (extracted_image, extracted_image_area_ratio), (first_image, first_image_area_ratio), (second_image, second_image_area_ratio)


def recursive_splitting(img, recursion_depth, area_ratio, image_index, search_threshold=0.6):
    global unique_counter
    #   Base cases
    if recursion_depth > 5:  # Safety to avoid infinite loop. 5 is more than enough
        return

    # in case one of the recursion returns None. Can happen in case of save or some other condition
    if img is None:
        return

    # Compute and store the save path once
    unique_save_path = output_save_path + '/' + \
        "{:04d}".format(image_index) + '_' + \
        "{:04d}".format(unique_counter)+'.jpg'

    # if the match covers this much area then stop recursion and just save the image
    if area_ratio > 0.7:  # Based on experiments

        plt.imsave(unique_save_path,
                   img, cmap='gray')
        unique_counter += 1
        return

    match_scores = get_match_scores(img, search_threshold)

    # return from recursion if the image has no matches
    if len(match_scores.values()) == 0:
        return

    # Save image if only one class is found
    if len(match_scores) == 1:

        plt.imsave(unique_save_path,
                   img, cmap='gray')
        unique_counter += 1
        return

    max_label = max(match_scores.items(), key=operator.itemgetter(1))[0]

    best_match_details = {}
    temp_image = img.copy()
    # Go through all the images of the class to do template matching and find the best match
    for path in glob.glob(character_image_path+max_label+'/*.pgm'):
        template = cv2.imread(path, 0)
        template_x, template_y = template.shape
        scale_ratio = orig_x/template_x
        if scale_ratio > 2:  # Experiments found this scale better
            scale_ratio = 2
        template_scores = []
        search_image_x, search_image_y = temp_image.shape
        # for scale in np.arange(0.2, scale_ratio, 0.1):
        y_ratio = (int)(scale_ratio*template_x)
        if y_ratio > search_image_x:
            y_ratio = search_image_x
        x_ratio = (int)(scale_ratio*template_y)
        if x_ratio > search_image_y:
            x_ratio = search_image_y
        if x_ratio <= 0 or y_ratio <= 0:
            continue
        template = cv2.resize(
            template, (x_ratio, y_ratio), interpolation=cv2.INTER_AREA)

        w, h = template.shape[::-1]
        try:
            res = cv2.matchTemplate(temp_image, template, method)
        except Exception as e:
            print("Error occured while processing")
            print(e)
            return
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        w, h = template.shape[::-1]
        best_match_details[max_val] = max_loc
    max_key = max(best_match_details.keys())
    top_left = best_match_details[max_key]
    (extracted_image, extracted_image_ar), (first_image, first_image_ar), (second_image,
                                                                           second_image_ar) = split_images(temp_image, top_left, h, w)

    #  Area ratio found based on experiments. Area ratio is found based on the original image.
    # If it ends up splitting upto this amount then assume that this region does not contain any character
    if extracted_image_ar < 0.2:
        extracted_image = None
    if first_image_ar < 0.2:
        first_image = None
    if second_image_ar < 0.2:
        second_image = None

    # Keep increasing threshold as image is broken down
    recursive_splitting(first_image, recursion_depth + 1,
                        first_image_ar, image_index, search_threshold+0.01)
    recursive_splitting(extracted_image, recursion_depth + 1,
                        extracted_image_ar, image_index, search_threshold+0.01)
    recursive_splitting(second_image, recursion_depth + 1,
                        second_image_ar, image_index, search_threshold+0.01)


def iterate_over_characters(input_path, character_images_path):
    """
    This is the entry point for this module.
    args:
    input_path: path of the character images
    character_images_path: path to store the output
    """
    # some variables are declared global so that they can be shared without unnecessary parameters in the functions.
    # This needs to be called whenever a global variable is assigned in a function so that it does not make a local copy
    global categories
    global unique_counter
    global orig_x
    global orig_y
    global output_save_path
    global character_image_path
    character_image_path = character_images_path
    categories = os.listdir(character_image_path)
    start = time.process_time()
    # Iterate over each line
    for line_character_folders in os.listdir(os.path.join(input_path)):
        print("Computing for line", line_character_folders)
        # Iterate over each character in the image
        for input_image_path in tqdm.tqdm(glob.glob(os.path.join(input_path,  line_character_folders)+"/*.jpg")):
            input_character_image = cv2.imread(input_image_path, 0)
            orig_x, orig_y = input_character_image.shape
            # Based on experiments. If the ratio is greater then it is assumed that the segemented image is a combination of few characters
            if (orig_y/orig_x) < 1.2:
                continue
            unique_counter = 1
            output_save_path = os.path.join(
                input_path, line_character_folders)
            image_index = int(input_image_path.split('.')
                              [0].split('/')[-1])
            os.makedirs(output_save_path, exist_ok=True)
            # Split the character
            recursive_splitting(input_character_image, 1, 0, image_index)
            # Remove original character image as it has been split
            os.remove(input_image_path)
    print("Time taken", time.process_time() - start)
