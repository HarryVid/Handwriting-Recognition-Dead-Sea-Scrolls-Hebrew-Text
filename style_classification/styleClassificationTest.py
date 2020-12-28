import gc
import os
import warnings

warnings.filterwarnings("ignore")

import pickle
import numpy as np
import tensorflow as tf
from cv2 import cv2
from tqdm import tqdm
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

def styleClassification(path_to_char_recog_model, path_to_segmented_images, document_name):
    
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                      device_count={'GPU': 0, 'CPU': 10})
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    cr = load_model(path_to_char_recog_model)

    test = os.listdir(path_to_segmented_images+'/'+document_name)
    classes = []

    test_class = (
        "Alef", "Ayin", "Bet", "Dalet", "Gimel", "He", "Het", "Kaf", "Kaf-final", "Lamed", "Mem", "Mem-medial",
        "Nun-final",
        "Nun-medial", "Pe", "Pe-final", "Qof", "Resh", "Samekh", "Shin", "Taw", "Tet", "Tsadi-final", "Tsadi-medial",
        "Waw",
        "Yod", "Zayin")

    for a in tqdm(test):
        image = os.listdir(path_to_segmented_images+'/'+document_name + "/" + str(a))
        for b in image:
            testimages = []
            testimg = cv2.imread(path_to_segmented_images+'/'+document_name + "/" + str(a) + "/" + str(b), cv2.IMREAD_GRAYSCALE)
            img_height, img_width = testimg.shape[0], testimg.shape[1]
            testresized = cv2.resize(testimg, (32, 49), interpolation=cv2.INTER_LINEAR)
            testimages.append(testresized)
            testimages = np.array(testimages)
            testimages = testimages / 255.0
            testimages = testimages.reshape(testimages.shape[0], testimages.shape[1], testimages.shape[2], 1)
            if (img_height / img_width) > 0.23:
                y = cr.predict(testimages)
                confidence = np.max(y) * 100
                if confidence > 25:
                    character = np.argmax(cr.predict(testimages), axis=-1)
                    character = test_class[character[0]]
                    with open("style_classification/Models/" + str(character) + ".pkl", 'rb') as fid:
                        clf = pickle.load(fid)

                    testimages = []
                    testresized = cv2.resize(testimg, (50, 50), interpolation=cv2.INTER_LINEAR)
                    testimages.append(testresized)
                    testimages = np.array(testimages)
                    testimages = testimages / 255.0
                    testimages = testimages.reshape(testimages.shape[0], testimages.shape[1], testimages.shape[2], 1)

                    K.clear_session()

                    model = Sequential()
                    model.add(Input(shape=(50, 50, 1)))
                    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                    model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                    model.add(Convolution2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(Convolution2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(Convolution2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                    model.add(Convolution2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(Convolution2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(Convolution2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
                    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                    model.add(Flatten())

                    rescaledTestFeatures = model.predict(testimages)

                    ypred = clf.predict(rescaledTestFeatures)
                    classes.append(ypred[0])

            gc.collect()

    style = max(set(classes), key = classes.count)

    if style == 0:
        decoderLabel = "Archaic"
    elif style == 1:
        decoderLabel = "Hasmonean"
    elif style == 2:
        decoderLabel = "Herodian"

    f = open("output/"+str(document_name) + "StyleOutput.txt", "w")
    f.write(str(decoderLabel))
    f.close()
