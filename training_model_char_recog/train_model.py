import tensorflow as tf
import numpy as np
import os
import cv2
import random
from keras.utils import to_categorical
import time


# function to generate training data for feed forward model - alexnet
def make_data(data_dir, label_name):
    training_data = []
    X = []
    y = []
    for l in label_name:
        read_path = os.path.join(data_dir, l)
        class_num = label_name.index(l)
        for filename in os.listdir(read_path):
            img = cv2.imread(os.path.join(read_path, filename), cv2.IMREAD_GRAYSCALE)
            training_data.append([img, class_num])

    random.shuffle(training_data)

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, 49, 32, 1)
    y = np.array(y)
    X = X / 255.0
    # need to do to_categorical for loss categorical_crossentropy
    y = to_categorical(y)
    return X, y


# complie the alexnet model with Adam optimizer and learning rate 0.0005, having two extra layers
def compile_model(X, class_size, loss, hidden_activation):
    img_height, img_width, channel = X.shape[1], X.shape[2], X.shape[3]
    model = tf.keras.Sequential()

    #  First Layer
    model.add(tf.keras.layers.Conv2D(96, (11, 11), strides=(1, 1), padding="same", activation=hidden_activation,
                                     input_shape=(img_height, img_width, channel)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    # Second Layer
    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding="same", activation=hidden_activation))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    # Third Layer
    model.add(tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same", activation=hidden_activation))

    # Fourth Layer
    model.add(tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same", activation=hidden_activation))

    # Fifth Layer
    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation=hidden_activation))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    # Fifth Layer (extra)
    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation=hidden_activation))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    # Fifth Layer (one more extra)
    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation=hidden_activation))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    # Sixth Layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation=hidden_activation))

    # Seventh Layer
    model.add(tf.keras.layers.Dense(4096, activation=hidden_activation))
    model.add(tf.keras.layers.Dropout(0.25))

    # Output Layer
    model.add(tf.keras.layers.Dense(class_size, activation="softmax"))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    model.summary()
    return model


# run the model with early stopping 10% validation split
def run_model(model, X, y, epochs):
    NAME = f"HR_result_{int(time.time())}"
    my_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=f"logs/{NAME}"),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    ]
    model.fit(X, y, epochs=epochs, validation_split=0.1, callbacks=my_callbacks)
    model_name = f"HR_CR_model_{int(time.time())}.h5"
    model.save(model_name)


def main():
    label_name = (
        "Alef", "Ayin", "Bet", "Dalet", "Gimel", "He", "Het", "Kaf", "Kaf-final", "Lamed", "Mem", "Mem-medial",
        "Nun-final", "Nun-medial", "Pe", "Pe-final", "Qof", "Resh", "Samekh", "Shin", "Taw", "Tet", "Tsadi-final",
        "Tsadi-medial", "Waw", "Yod", "Zayin")

    loss = "categorical_crossentropy"
    hidden_activation = "relu"
    epochs = 50

    read_path = os.getcwd()
    read_path = os.path.split(read_path)[0]
    read_path = os.path.join(read_path, "character_images")
    data_dir = os.path.join(read_path, "Images_final_preprocess")

    class_size = len(label_name)
    X, y = make_data(data_dir, label_name)

    alex_net = compile_model(X, class_size, loss, hidden_activation)
    run_model(alex_net, X, y, epochs)


if __name__ == "__main__":
    main()
