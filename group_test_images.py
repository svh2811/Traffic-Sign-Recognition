from constants import *

import os
import csv

for i, class_folder in enumerate(os.listdir(training_dir)):
    target_test_class_directory = test_dir + "/" + class_folder
    if not os.path.exists(target_test_class_directory):
        print("Creating Dir: ", target_test_class_directory)
        os.makedirs(target_test_class_directory)

with open(base_directory + "/GT-final_test.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for i, row in enumerate(readCSV):
        if i != 0:
            source = test_dir + "/" + row[0]
            target = test_dir + "/" + "{:05d}".format(int(row[7])) + "/" + row[0]
            # print(source, target)
            os.rename(source, target)
            # break
