import pandas as pd
import tensorflow as tf
import pandas as pd
import numpy as np
import nibabel as nib
from utils.constants import Constants

class RNNData:
    def __init__(self, args):
        self.pathway = "../data/adni.csv"
        self.dof = "9DOF"
        self.target_column = "label"
        self.batch_size = args.batch_size
        self.args = args

        self.total_df = pd.DataFrame()
        self.train_df = pd.read_csv("/project/jambitem_1194/pjtoral/data/adni_train.csv")
        self.validation_df = pd.read_csv("/project/jambitem_1194/pjtoral/data/adni_validation.csv")
        self.test_df = pd.read_csv("/project/jambitem_1194/pjtoral/data/adni_test.csv")
        self.train_batch = pd.DataFrame()
        self.validation_batch = pd.DataFrame()
        self.test_batch = pd.DataFrame()
        self.set_data_generators()
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

    def create_sequence_and_target(self, df):
        sequence_list = []
        target_class_list = []
        unique_subj_ids = df["subj_id"].unique()
        for subj_id in unique_subj_ids:
            # Retrieve a Subject Sequence
            subject_sequence = df[df["subj_id"] == subj_id]
            # Chronologically Order the subject sequence
            subject_sequence_chronologically_ordered = subject_sequence.sort_values(by=["subj_id", "viscode"])

            if self.args.sequence_length is not None:
                seq_len = self.args.sequence_length
                subj_seq_len = subject_sequence_chronologically_ordered.shape[0]
                if subj_seq_len >= seq_len:
                    if seq_len == 1:
                        for i in range(subject_sequence_chronologically_ordered["volume"].shape[0]):
                            pathway = subject_sequence_chronologically_ordered["volume"].values[i]
                            sequence_list.append([pathway])
                        for label in subject_sequence_chronologically_ordered["label"]:
                            target_class_list.append([label])
                    else:
                        # set sequence testing
                        i = 0
                        j = (i + seq_len - 1)
                        working_pathway_list = []
                        working_label_list = []
                        while j < subj_seq_len:
                            working_pathway_list.extend(subject_sequence_chronologically_ordered["volume"].values[i:j])
                            for label in subject_sequence_chronologically_ordered["label"].values[i:j]:
                                working_label_list.append(label)
                            i += 1
                            j = (i + seq_len - 1)
                        target_class_list.append(working_label_list)
                        sequence_list.append(working_pathway_list)
            else:
                # Arbitrary Sequence Length
                label_array = []
                for label in subject_sequence["label"]:
                    label_array.append(label)
                sequence_list.append(subject_sequence["volume"].values)
                target_class_list.append(label_array)
        return sequence_list, target_class_list

    def set_data_generators(self):
        if self.args.task_selection == Constants.binary_classification:
            self.set_binary_classification_labels()
        elif self.args.task_selection == Constants.multi_classification:
            self.set_multi_classification_labels()

        train_sequence_list, train_target_class_list = self.create_sequence_and_target(self.train_df)
        validation_sequence_list, validation_target_class_list = self.create_sequence_and_target(self.validation_df)
        test_sequence_list, test_target_class_list = self.create_sequence_and_target(self.test_df)

        print("\nTrain Sequences:", len(train_sequence_list), ", Target Classes:", len(train_target_class_list))
        print("Validation Sequences:", len(validation_sequence_list), ", Target Classes:"
              , len(validation_target_class_list))
        print("Test Sequences:", len(test_sequence_list), ", Target Classes:", len(test_target_class_list))

        self.train_batch = self.RNNDataGenerator(train_sequence_list, train_target_class_list, self.batch_size)
        self.validation_batch = self.RNNDataGenerator(validation_sequence_list, validation_target_class_list
                                                       , self.batch_size)
        self.test_batch = self.RNNDataGenerator(test_sequence_list, test_target_class_list, self.batch_size)

    class RNNDataGenerator(tf.keras.utils.Sequence):
        def read_scan(self, path):
            scan = nib.load(path)
            original_volume = scan.get_fdata()
            original_volume_normalized = self.normalize(original_volume)
            return tf.expand_dims(original_volume_normalized, axis=3)

        def normalize(self, volume):
            min = np.amax(volume)
            max = np.amin(volume)
            volume = (volume - min) / (max - min)
            volume = volume.astype("float32")
            return volume

        def __init__(self, image_filenames, labels, batch_size, sample_weights=None):
            self.image_filenames = image_filenames
            self.labels = labels
            self.batch_size = batch_size

        def __len__(self):
            return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int32)

        def __getitem__(self, idx):
            batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
            batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
            output_batch_x = np.asarray([self.read_scan(path) for path in batch_x[0]])
            output_batch_y = []

            for y in batch_y:
                out_y = np.asarray(y)
                output_batch_y.append(out_y)

            # for y in batch_y[0]:
            #   out_y = np.asarray(y).reshape(1,3)
            # output_batch_y = np.asarray(batch_y[0]).reshape(1,3)
            # print(output_batch_y)
            item = (
                output_batch_x.reshape(self.batch_size, output_batch_x.shape[0], output_batch_x.shape[1]
                                       , output_batch_x.shape[2], output_batch_x.shape[3], output_batch_x.shape[4])
                , np.asarray(output_batch_y)
            )
            return item
