import cv2
import os
import pickle
import numpy as np


class CovidImage:

    def __init__(self, path, train_file, test_file, classes=['normal', 'pneumonia', 'COVID-19'], img_size=224):
        self.path = path
        self.train_file = train_file
        self.test_file = test_file
        self.img_size = img_size

        self.train_images = self._process_csv_file(train_file)
        self.train_dir = os.path.join(path, "train")
        self.train_N = len(self.train_images)
        self.trainX = []
        self.train_y = []

        self.test_images = self._process_csv_file(test_file)
        self.test_dir = os.path.join(path, "test")
        self.test_N = len(self.test_images)
        self.testX = []
        self.test_y = []

        self.classes = classes
        self.class_id = {c: classes.index(c) for c in classes}
        for c in classes:
            self.class_id[c+"\n"] = classes.index(c)

    def load_data(self):

        try:
            self._load_pickled_train_data()
            print('Loading training data')

        except:
            print('Could not find pickled traning data')
            self._process_train_set()

        try:
            self._load_pickled_test_data()
            print('Loading test data')
        except:
            print('Could not find pickled test data')
            self._process_test_set()

        print("Loading done")

        return self.trainX, self.train_y, self.testX, self.test_y

    def _process_csv_file(self, file):
        with open(file, 'r') as fr:
            files = fr.readlines()
        return files

    def _process_train_set(self):
        print("Processing train set")
        image_data = []
        c = 0

        for datapoint in self.train_images:
            sample = datapoint.split(" ")
            img = sample[1]
            label = sample[2]
            c = c + 1
            if c % 1000 == 0:
                print("Processed {} images of {}".format(c, self.train_N))
            img_path = os.path.join(self.train_dir, img)
            # try:  # if any image is corrupted
            image_data_temp = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read Image as numbers
            image_temp_resize = cv2.resize(image_data_temp, (self.img_size, self.img_size))
            image_data.append([image_temp_resize, self.class_id[label]])
            # except:
            #     pass

        data = np.asanyarray(image_data)
        X = []
        y = []
        # Iterate over the Data
        for x in data:
            X.append(x[0])
            y.append(x[1])

        self.trainX = np.asarray(X)/(255.0)
        self.train_y = np.asarray(y)

        self._pickle_train_data()
        print("Done with train data")

    def _process_test_set(self):
        print("Processing test set")
        image_data = []
        c = 0

        for datapoint in self.test_images:
            sample = datapoint.split(" ")
            img = sample[1]
            label = sample[2]
            c = c + 1
            if c % 1000 == 0:
                print("Processed {} images of {}".format(c, self.test_N))

            img_path = os.path.join(self.test_dir, img)
            # try:
            image_data_temp = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read Image as numbers
            image_temp_resize = cv2.resize(image_data_temp, (self.img_size, self.img_size))
            image_data.append([image_temp_resize, self.class_id[label]])
            # except:
            #     pass

        data = np.asanyarray(image_data)

        X = []
        y = []
        for x in data:
            X.append(x[0])
            y.append(x[1])

        self.testX = np.asarray(X)/(255.0)
        self.test_y = np.asarray(y)

        self._pickle_test_data()
        print("Done with test data")

    def _pickle_train_data(self):
        N = self.trainX.shape[0]
        N2 = int(N/2)
        with open(os.path.join(self.path, "trainX1.pickle"), "wb") as f:
            pickle.dump(self.trainX[0:N2, :, :], f)

        with open(os.path.join(self.path, "trainX2.pickle"), "wb") as f:
            pickle.dump(self.trainX[N2:2, :, :], f)

        with open(os.path.join(self.path, "train_y.pickle"), "wb") as f:
            pickle.dump(self.train_y, f)

    def _pickle_test_data(self):
        with open(os.path.join(self.path, "testX.pickle"), "wb") as f:
            pickle.dump(self.testX, f)

        with open(os.path.join(self.path, "test_y.pickle"), "wb") as f:
            pickle.dump(self.test_y, f)

    def _load_pickled_train_data(self):
        with open(os.path.join(self.path, "trainX1.pickle"), "rb") as f:
            X1 = pickle.load(f)

        with open(os.path.join(self.path, "trainX2.pickle"), "rb") as f:
            X2 = pickle.load(f)

        self.trainX = np.concatenate((X1, X2))

        with open(os.path.join(self.path, "train_y.pickle"), "rb") as f:
            self.train_y = pickle.load(f)

    def _load_pickled_test_data(self):
        with open(os.path.join(self.path, "testX.pickle"), "rb") as f:
            self.testX = pickle.load(f)

        with open(os.path.join(self.path, "test_y.pickle"), "rb") as f:
            self.test_y = pickle.load(f)


def main():

    path = "data"
    train_file = "data/train_split_v3.txt"
    test_file = "data/test_split_v3.txt"

    dataloader = CovidImage(path, train_file, test_file)
    trainX, train_y, testX, test_y = dataloader.load_data()



if __name__ == "__main__":
    main()
