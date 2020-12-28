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
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential

config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                  device_count={'GPU': 0, 'CPU': 10})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

characters = os.listdir(".//styleClassificationTrain//")

for character in tqdm(characters):
    periods = os.listdir(".//styleClassificationTrain//" + str(character))

    images = []
    labels = []

    for x in periods:
        image = os.listdir(".//styleClassificationTrain//" + str(character) + "//" + str(x))
        for y in image:
            img = cv2.imread(".//styleClassificationTrain//" + str(character) + "//" + str(x) + "//" + str(y), cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
            images.append(resized)
            if x == "Archaic":
                labels.append(0)
            elif x == "Hasmonean":
                labels.append(1)
            elif x == "Herodian":
                labels.append(2)

    images = np.array(images)
    images = images / 255.0
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    encoderLabels = np.array(labels)

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

    rescaledFeatures = model.predict(images)

    gnb = GaussianNB()
    gnb.fit(rescaledFeatures, encoderLabels)

    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(rescaledFeatures, encoderLabels)

    ert =ExtraTreesClassifier(n_jobs=-1)
    ert.fit(rescaledFeatures, encoderLabels)

    adb = AdaBoostClassifier()
    adb.fit(rescaledFeatures, encoderLabels)

    gb = GradientBoostingClassifier()
    gb.fit(rescaledFeatures, encoderLabels)

    hgb = HistGradientBoostingClassifier()
    hgb.fit(rescaledFeatures, encoderLabels)

    classifiers = [("gnb", gnb), ("rf", rf), ("ert", ert), ("adb", adb), ("gb", gb), ("hgb", hgb)]
    clf = StackingClassifier(estimators=classifiers, n_jobs=-1, cv=2, passthrough=True)
    clf.fit(rescaledFeatures, encoderLabels)
    print(clf.score(rescaledFeatures, encoderLabels))

    with open(".//Models//" + str(character) + ".pkl", 'wb') as fid:
        pickle.dump(clf, fid)

    gc.collect()
