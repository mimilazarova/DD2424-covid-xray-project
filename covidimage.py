import tensorflow as tf
import cv2
import os
import pickle
import numpy as np



class CovidImage:

    def __init__(self, path, train_file, test_file, classes=['normal', 'pneumonia', 'covid19'], img_size=224):
        self.path = path
        self.train_file = train_file
        self.test_file = test_file
        self.img_size = img_size

        self.train_images = self._process_csv_file(train_file)
        self.train_dir = os.path.join(path, "train")
        self.trainX = []
        self.train_y = []

        # with open(test_file, 'rb') as f:
        #     self.test_images = f.readlines()
        self.test_images = self._process_csv_file(test_file)
        self.test_dir = os.path.join(path, "test")
        self.testX = []
        self.test_y = []

        self.classes = classes
        self.class_id = {c: classes.index(c) for c in classes}

    def _process_csv_file(self, file):
        with open(file, 'r') as fr:
            files = fr.readlines()
        return files

    def _process_train_set(self):
        image_data = []
        for datapoint in self.train_images:
            sample = datapoint.split(" ")
            img = sample[1]
            label = sample[2]

            img_path = os.path.join(self.train_dir, img)
            try:  # if any image is corrupted
                image_data_temp = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read Image as numbers
                image_temp_resize = cv2.resize(image_data_temp, (self.img_size, self.img_size))
                image_data.append([image_temp_resize, self.class_id(label)])
            except:
                pass

            data = np.asanyarray(image_data)

            # Iterate over the Data
            for x in data:
                self.trainX.append(x[0])
                self.train_y.append(x[1])

            # X_Data = np.asarray(self.x_data) / (255.0)
            # Y_Data = np.asarray(self.y_data)

def main():

    path = "data"
    train_file = "data/train_split_v3.txt"
    test_file = "data/test_split_v3.txt"

    dataloader = CovidImage(path, train_file, test_file)

    print("yey")
