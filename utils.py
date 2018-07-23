import os
import numpy as np
import csv
import tensorflow as tf
from skimage import io
from sklearn.model_selection import train_test_split


_NUM_CLASSES = None
_IMG_SHAPE = None
_BATCH_SIZE = None

def init_global_vars(num_class, img_shape, batch):
    global _NUM_CLASSES
    _NUM_CLASSES = num_class
    global _IMG_SHAPE
    _IMG_SHAPE = img_shape
    global _BATCH_SIZE
    _BATCH_SIZE = batch


def get_examples_dataset():
    database = {
        "X" : [],
        "y" : []
    }
    base_dir = "./examples"

    for imgName in os.listdir(base_dir):
        database["X"].append(os.path.join(base_dir, imgName))
        database["y"].append(int(imgName[0:2]))

    return database


def get_traffic_sign_dataset(test_ratio = 0.10):
    directory = "./data"
    training_dir = directory + "/GTSRB/Final_Training/Images/"
    testing_dir = directory + "/GTSRB/Final_Test/Images/"

    dataset = {
        "train" : {},
        "val" : {},
        "test" : {}
    }

    X = []
    y = []

    for i, class_folder in enumerate(os.listdir(training_dir)):
        class_folder_dir = training_dir + class_folder
        for imgName in os.listdir(class_folder_dir):
            index = imgName.index(".") + 1
            if imgName[index:] == "csv":
                continue
            X.append(os.path.join(class_folder_dir, imgName))
            y.append(int(class_folder))

    X_train, X_val, y_train, y_val =\
        train_test_split(X, y, test_size = test_ratio, shuffle = True)

    dataset["train"]["X"] = X_train
    dataset["train"]["y"] = y_train
    dataset["val"]["X"] = X_val
    dataset["val"]["y"] = y_val

    X_test = []
    y_test = []

    with open(os.path.join(directory, "GT-final_test.csv")) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(readCSV):
            if i == 0:
                continue
            X_test.append(os.path.join(testing_dir, row[0]))
            y_test.append(int(row[7]))
    dataset["test"]["X"] = X_test
    dataset["test"]["y"] = y_test

    return dataset


"""
Image: np.array of shape (W, H, C)
"""
def norm_image(image):
    image = image.astype(float)
    image = image[:, :, 0:3] # ignore alpha channel
    H, W, C = image.shape
    for i in np.arange(C):
        channel = image[:,:, i]
        min = channel.min()
        max = channel.max()
        image[:,:, i] = (image[:,:, i] - min) * (255.0 - 0.0)/(max - min) + 0.0
    return image


def _read_py_function(filename, label):
    # imread(...) returns a np.array of dtype int
    image_decoded_np = norm_image(io.imread(filename.decode()))
    return image_decoded_np, label


def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, None])
    image_resized = tf.image.resize_images(image_decoded,
                        [_IMG_SHAPE[0], _IMG_SHAPE[1]])
    label = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES)
    return image_resized, label


"""
dataset is a dictionary containing keys X and y | X, y are python list
"""
def covert_to_tf_dataset(dataset, batch = True, repeat = True):
    dataset_tf = tf.data.Dataset.from_tensor_slices((dataset["X"], dataset["y"]))
    dataset_tf = dataset_tf.map(
        lambda filename, label: tuple(tf.py_func(
            _read_py_function, [filename, label], [tf.float64, label.dtype])))
    dataset_tf = dataset_tf.map(_resize_function)
    if batch:
        dataset_tf = dataset_tf.batch(_BATCH_SIZE)
    if repeat:
        dataset_tf = dataset_tf.repeat()
    return dataset_tf


def get_class_name_dict():
    class_name_dict = {}
    with open(os.path.join("./data", "signnames.csv")) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(readCSV):
            if i == 0:
                continue
            class_name_dict[int(row[0])] = row[1]
    return class_name_dict
