import cv2
import os
import statistics


# function creates new images per class which are having same dimensions within those classes
def avg_dim(read_path):
    height = []
    width = []
    # iterate through each files append height and width then take average
    for filename in os.listdir(read_path):
        #if filename.endswith(".jpg"):
        img = cv2.imread(f"{read_path}/{filename}", -1)
        height.append(int(img.shape[0]))
        width.append(int(img.shape[1]))
    w = int(statistics.median(width))
    h = int(statistics.median(height))
    return w, h


# functions that saves the resized image in the folder "Images_first_preprocess"
def save_first_preprocess(r_folder, s_folder, l_name):
    for l in l_name:
        read_path = os.path.join(r_folder, l)
        # dim will have width and height
        dim = avg_dim(read_path)
        os.mkdir(f"{s_folder}/{l}")
        i = 0
        for filename in os.listdir(read_path):
            #if filename.endswith(".jpg"):
            i = i + 1
            # read image
            img = cv2.imread(f"{read_path}/{filename}", -1)
            # resize with the dim
            img = cv2.resize(img, dim)
            save_path = os.path.join(s_folder, l)
            # write the image
            cv2.imwrite(f"{save_path}/{l}_{i}.jpg", img)


# functions that saves the resized image in the folder "Images_final_preprocess"
def avgofavg_dim(r_folder, l_name):
    height = []
    width = []
    # iterate through each character first file append height and width then take average
    for l in l_name:
        read_path = os.path.join(r_folder, l)
        filename = os.listdir(read_path)[0]
        img = cv2.imread(f"{read_path}/{filename}", -1)
        height.append(int(img.shape[0]))
        width.append(int(img.shape[1]))

    # To preserve the shape of the rectangular images, we subtracted stdev in width and added stdev to height
    w = int(statistics.median(width)) - int(statistics.stdev(width))
    h = int(statistics.median(height)) + int(statistics.stdev(height))
    return w, h


# functions that saves the resized image in the folder "Images_final_preprocess", This result in all images having
# equal dimensions
def save_final_preprocess(r_folder, s_folder, l_name):
    # dim will have width and height
    dim = avgofavg_dim(r_folder, l_name)
    # iterate through each classes and each file and resize to the dim
    for l in l_name:
        read_path = os.path.join(r_folder, l)
        os.mkdir(f"{s_folder}/{l}")
        i = 0
        # iterate through each files
        for filename in os.listdir(read_path):
            #if filename.endswith(".jpg"):
            i = i + 1
            img = cv2.imread(f"{read_path}/{filename}", -1)
            img = cv2.resize(img, dim)
            save_path = os.path.join(s_folder, l)
            cv2.imwrite(f"{save_path}/{l}_{i}.jpg", img)


def delete_dir(folder):
    if os.path.exists(folder):
        os.rmdir(folder)


def create_dir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def main():
    # classes names
    label_name = (
        "Alef", "Ayin", "Bet", "Dalet", "Gimel", "He", "Het", "Kaf", "Kaf-final", "Lamed", "Mem", "Mem-medial",
        "Nun-final", "Nun-medial", "Pe", "Pe-final", "Qof", "Resh", "Samekh", "Shin", "Taw", "Tet", "Tsadi-final",
        "Tsadi-medial", "Waw", "Yod", "Zayin")

    read_path = os.getcwd()
    read_path = os.path.split(read_path)[0]
    read_path = os.path.join(read_path, "character_images")
    read_folder = os.path.join(read_path, "Images")
    save_folder = os.path.join(read_path, "Images_first_preprocess")
    # if folder does exit then delete "Images_first_preprocess" for final pre-processing
    delete_dir(save_folder)
    # if folder does not exit then create "Images_first_preprocess" for final pre-processing
    create_dir(save_folder)
    print(f"1 read_folder: {read_folder}")
    print(f"1 save_folder: {save_folder}")
    save_first_preprocess(read_folder, save_folder, label_name)

    # ------------------------------------------------------------------------------------------------------------------

    read_path = os.getcwd()
    read_path = os.path.split(read_path)[0]
    read_path = os.path.join(read_path, "character_images")
    read_folder = os.path.join(read_path, "Images_first_preprocess")
    save_folder = os.path.join(read_path, "Images_final_preprocess")
    # if folder does exit then delete "Images_final_preprocess" for final pre-processing
    delete_dir(save_folder)
    # if folder does not exit then create "Images_final_preprocess" for final pre-processing
    create_dir(save_folder)
    print(f"2 read_folder: {read_folder}")
    print(f"2 save_folder: {save_folder}")
    save_final_preprocess(read_folder, save_folder, label_name)


if __name__ == "__main__":
    main()
