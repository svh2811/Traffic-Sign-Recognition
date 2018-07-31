import os
import numpy as np
import csv
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from constants import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

import cv2
import itertools

def get_dataset(dir, save_img = False):
    dataset = {}
    X = []
    y = []

    for i, class_folder in enumerate(os.listdir(dir)):
        class_folder_dir = dir + "/" + class_folder
        for imgName in os.listdir(class_folder_dir):
            index = imgName.index(".") + 1
            if imgName[index:] == "csv":
                continue
            if save_img:
                X.append(os.path.join(class_folder_dir, imgName))
            y.append(int(class_folder))

    if save_img:
        dataset["X"] = X

    dataset["y"] = y

    return dataset


"""
Reference:
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
def balance_training_dataset(train_db_class_count, max_class_count):
    augment_count = max_class_count - train_db_class_count

    log = "class #{:03d} has {:04d} images"\
            + " {:04d} images approx will be added"
    for class_num, class_augment_count in enumerate(augment_count):
        print(log.format(class_num,
                        train_db_class_count[class_num],
                        class_augment_count))

        if class_augment_count == 0:
            continue

        target_class_dir = training_dir + "/{:05d}".format(class_num)

        """
        every class-folder is composed of several tracks and each track
        has exactly 30* snapshots, however the number of tracks
        in each class-folder would vary. Our approach is to randomly
        select an track and use 29th (0 Indexed) snapshot as it has
        highest resolution.
        """

        for _ in np.arange(class_augment_count):

            """
            This function may be called again on an already augmented
            dataset, hence it is imperative to know the original number of
            training images
            """
            count = 0
            csvFileName = target_class_dir + "/GT-{:05d}.csv".format(class_num)
            with open(csvFileName) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for i, _ in enumerate(readCSV):
                    if i != 0:
                        count += 1

            random_track = np.random.randint(count // 30)

            """
            Some rogue tracks have < 30 snapshots
            Eg: class 33 track 19 has no 29th snap
            """
            snap = 29
            file = None
            img = None
            while snap >= 0:
                try:
                    img = load_img(target_class_dir + "/{:05d}_{:05d}.ppm"\
                                                    .format(random_track, snap))
                    break
                except FileNotFoundError:
                    snap -=  1

            if snap == -1:
                print("class #{:03d} could not be augmented".format(class_num))
                break

            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)

            image_data_iterator = ImageDataGenerator(
                rotation_range = 30,
                shear_range = 0.30,
                width_shift_range = 0.30,
                height_shift_range = 0.30
            ).flow(img,
                batch_size = 1,
                save_to_dir = target_class_dir,
                save_prefix = "keras_aug")

            # augment one image at a time
            for batch in image_data_iterator:
                break


def get_class_name_dict():
    class_name_dict = {}
    with open(base_directory + "/signnames.csv") as csvfile:
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
    plt.figure(figsize = (20, 20))
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

    plt.rcParams["figure.figsize"] = [14, 6]
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


def draw_line_graphs(y1, y1_label = "",
                    y2 = None, y2_label = "",
                    title = "", xlabel = "", ylabel = "",
                    legend_loc = "best"):

    plt.figure(figsize = (16, 9))
    plt.tight_layout()

    x_s = np.arange(len(y1))
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


# f1 score punishes extreme values
def get_f1_score(precision, recall):
    return 2.0 * (precision * recall) / (precision + recall)


# image is a grayscale image of shape (H, W, 1)
def histogram_equalize_rgb_image(image):
    # equalizeHist requires int dtype
    y = cv2.equalizeHist(np.array(image, dtype = np.uint8))
    y = y.reshape((y.shape[0], y.shape[1], 1))
    return y.astype(np.float32)

"""
else:

    if C > 3:
        image = image[:, :, :3]
    img_ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB) # float32 image
    y = np.array(img_ycbcr[:, :, 0], dtype = np.uint8)
    y = cv2.equalizeHist(y) # equalizeHist requires int dtype
    img_ycbcr[:, :, 0] = y.astype(np.float32)
    return cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2RGB)
"""


def tensor_to_numpy(tensor):
    # tensor.numpy() only works in eager execution mode
    sess = tf.Session()
    with sess.as_default():
        numpy_arr = tensor.eval()
    return numpy_arr
