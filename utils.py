import os
import numpy as np
import csv
import tensorflow as tf
from skimage import io
from sklearn.model_selection import train_test_split

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path


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
    dataset_tf = tf.data.Dataset.from_tensor_slices(
                    (dataset["X"], dataset["y"]))
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


"""
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (13, 13))
    plt.tight_layout()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix")
    else:
        print("\nConfusion matrix, without normalization")

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()
    plt.close()


"""
https://matplotlib.org/gallery/api/histogram_path.html#sphx-glr-gallery-api-histogram-path-py
"""
def plot_bar_chart(bar_heights, title = "",
                    xlabel = "", ylabel = "",
                    fileName = None):

    plt.rcParams["figure.figsize"] = [16, 9]
    plt.tight_layout()

    fig, ax = plt.subplots()
    bins = np.arange(bar_heights.shape[0] + 1)

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + bar_heights

    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(barpath)
    ax.add_patch(patch)

    width = right[0] - left[0]
    # Hide major tick labels
    ax.set_xticklabels('')

    # Customize minor tick labels
    ax.set_xticks(bins + width / 2.0, minor=True)
    ax.set_xticklabels(bins, minor=True)
    ax.set_xticks(bins)

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.grid(True)

    if fileName is None:
        plt.show()
    else:
        fig.savefig(fileName, bbox_inches = 'tight')
        # closing the plot is important or else the any previous plot's
        # will be shown in output

    plt.close()


def plot_histogram(bars, title,
                    xlabel = "Class Number",
                    ylabel = "Class Count"):
    bar_heights = np.bincount(bars, minlength = _NUM_CLASSES)
    plot_bar_chart(bar_heights, title, xlabel, ylabel)


def draw_line_graphs(x_max, y1, y1_label = "",
                    y2 = None, y2_label = "",
                    title = "", xlabel = "", ylabel = "",
                    legend_loc = "best"):

    plt.figure(figsize = (16, 9))
    plt.tight_layout()

    x_s = np.arange(x_max)
    plt.plot(x_s, y1, 'b.-', label = y1_label)
    if y2 is not None:
        plt.plot(x_s, y2, 'r.-', label = y2_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    leg = plt.legend(loc = legend_loc)
    leg.get_frame().set_alpha(0.5)
    plt.grid(True)
    plt.show()
    plt.close()


def get_conf_matrix_stat(conf_matrix):
    row_sum = conf_matrix.sum(axis = 1)
    col_sum = conf_matrix.sum(axis = 0)
    tp = conf_matrix.diagonal()
    tn = conf_matrix.sum() - row_sum - col_sum + tp
    fn = row_sum - tp
    fp = col_sum - tp
    return tp, fp, tn, fn


def get_classfier_stats(tp, fp, tn, fn):
    precision = np.nan_to_num(tp/(tp + fp))

    recall = np.nan_to_num(tp/(tp + fn))
    # or sensitivity or true positive rate

    specificity = np.nan_to_num(tn/(tn + fp))
    # or true negative rate

    return precision, recall, specificity
