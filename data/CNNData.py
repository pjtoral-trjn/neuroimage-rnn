import pandas as pd
import numpy as np
import nibabel as nib
import tensorflow as tf
from utils.constants import Constants

class CNNData:
    def __init__(self, args):
        self.args = args
        self.pathway = "/lfs1/pjtoral/cognitive-decline/scripts/data/revised/standardized/mci_included"
        self.dof = "9DOF"
        self.target_column = self.args.target_column
        self.batch_size = self.args.batch_size

        self.train_df = pd.DataFrame()
        self.validation_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.train_batch = pd.DataFrame()
        self.validation_batch = pd.DataFrame()
        self.test_batch = pd.DataFrame()
        self.set_dataframes()
        self.set_data_generators()

    def set_dataframes(self):
        df_train = pd.read_csv("/project/jambitem_1194/pjtoral/data/adni_train.csv")
        df_validation = pd.read_csv("/project/jambitem_1194/pjtoral/data/adni_validation.csv")
        df_test = pd.read_csv("/project/jambitem_1194/pjtoral/data/adni_test.csv")

        df_train.dropna(subset=[self.target_column], inplace=True)
        df_validation.dropna(subset=[self.target_column], inplace=True)
        df_test.dropna(subset=[self.target_column], inplace=True)

        self.train_df = df_train
        self.validation_df = df_validation
        self.test_df = df_test

    def set_binary_classification_labels(self):
        # print(self.train_df["label"].value_counts())
        # print(self.validation_df["label"].value_counts())
        # print(self.test_df["label"].value_counts())

        # Removing 1 = MCI for binary classification
        self.train_df.drop(self.train_df.loc[self.train_df['label'] == 1].index, inplace=True)
        self.validation_df.drop(self.validation_df.loc[self.validation_df['label'] == 1].index, inplace=True)
        self.test_df.drop(self.test_df.loc[self.test_df['label'] == 1].index, inplace=True)

        # Converting Class to Binary
        train_index = self.train_df.loc[self.train_df['label'] == 2].index
        self.train_df.loc[train_index, 'label'] = 1

        validation_index = self.validation_df.loc[self.validation_df['label'] == 2].index
        self.validation_df.loc[validation_index, 'label'] = 1

        test_index = self.test_df.loc[self.test_df['label'] == 2].index
        self.test_df.loc[test_index, 'label'] = 1

        # print(self.train_df["label"].value_counts())
        # print(self.validation_df["label"].value_counts())
        # print(self.test_df["label"].value_counts())

    def set_multi_classification_labels(self):
        arr = []
        for i in range(len(self.train_df["label"].values)):
            # print("CN vs MCI vs AD")
            label_ = self.train_df["label"].values[i]
            if label_ == 0:
                label = [1, 0, 0]
            elif label_ == 1:
                label = [0, 1, 0]
            elif label_ == 2:
                label = [0, 0, 1]
            arr.append(np.asarray(label))
        self.train_df["label"] = arr

        arr = []
        for i in range(len(self.validation_df["label"].values)):
            # print("CN vs MCI vs AD")
            label_ = self.validation_df["label"].values[i]
            if label_ == 0:
                label = [1, 0, 0]
            elif label_ == 1:
                label = [0, 1, 0]
            elif label_ == 2:
                label = [0, 0, 1]
            arr.append(np.asarray(label))
        self.validation_df["label"] = arr

        arr = []
        for i in range(len(self.test_df["label"].values)):
            # print("CN vs MCI vs AD")
            label_ = self.test_df["label"].values[i]
            if label_ == 0:
                label = [1, 0, 0]
            elif label_ == 1:
                label = [0, 1, 0]
            elif label_ == 2:
                label = [0, 0, 1]
            arr.append(np.asarray(label))
        self.test_df["label"] = arr


    def set_data_generators(self):
        if self.args.task_selection == Constants.binary_classification:
            self.set_binary_classification_labels()
        elif self.args.task_selection == Constants.multi_classification:
            self.set_multi_classification_labels()

        train_x = self.train_df["volume"].to_numpy()
        #train_y = self.train_df[self.target_column].to_numpy().astype(np.float32)
        train_y = self.train_df[self.target_column].to_numpy()
        validate_x = self.validation_df["volume"].to_numpy()
        #validate_y = self.validation_df[self.target_column].to_numpy().astype(np.float32)
        validate_y = self.validation_df[self.target_column].to_numpy()
        test_x = self.test_df["volume"].to_numpy()
        #test_y = self.test_df[self.target_column].to_numpy().astype(np.float32)
        test_y = self.test_df[self.target_column].to_numpy()

        self.train_batch = self.DataGenerator(train_x, train_y, self.batch_size, self.args)
        self.validation_batch = self.DataGenerator(validate_x, validate_y, self.batch_size, self.args)
        self.test_batch = self.DataGenerator(test_x, test_y, self.batch_size, self.args)

    class DataGenerator(tf.keras.utils.Sequence):
        def read_scan(self, path):
            scan = nib.load(path)
            original_volume = scan.get_fdata()
            original_volume_normalized = self.normalize(original_volume)
            # resized_volume = self.resize(original_volume_normalized)
            return tf.expand_dims(original_volume_normalized, axis=3)

        def normalize(self, volume):
            min = np.amax(volume)
            max = np.amin(volume)
            volume = (volume - min) / (max - min)
            volume = volume.astype("float32")
            return volume


        def __init__(self, image_filenames, labels, batch_size, args):
            self.image_filenames = image_filenames
            self.labels = labels
            self.batch_size = batch_size
            self.args = args

        def __len__(self):
            return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int32)

        def __getitem__(self, idx):
            batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
            batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
            if self.args.task_selection == Constants.multi_classification:
                return (np.asarray([self.read_scan(path) for path in batch_x]), np.asarray(batch_y[0]).reshape((1,3)))
            return (np.asarray([self.read_scan(path) for path in batch_x]), np.asarray(batch_y))