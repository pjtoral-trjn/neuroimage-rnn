import tensorflow as tf
import os
import sys
import pandas as pd
import datetime
from model.rnn import rnn
from utils.constants import Constants
from data.RNNData import RNNData

class Pipeline:
    def __init__(self, args):
        self.optimizer = None
        self.history = None
        self.test_batch = None
        self.validation_batch = None
        self.train_batch = None
        self.data = None
        self.metrics = None
        self.callbacks = None
        self.loss_fn = None
        self.model = None

        self.args = args
        self.creation_time_for_csv_output = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.output_filename = self.args.experiment_name + "__" + self.creation_time_for_csv_output
        self.best_weights_checkpoint_filepath = './model_checkpoint/' + self.args.target_column + "/" \
                                                + self.creation_time_for_csv_output + '/checkpoint'

    def configure_gpu(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
        physical_devices = tf.config.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(physical_devices), ", GPU ID: ", str(self.args.gpu))
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def configure_model(self):
        self.model = rnn(self.args)
        if self.model is not None:
            self.set_optimizer()
            self.set_loss_fn()
            self.set_callbacks()
            self.set_metrics()
            self.compile()

    def set_optimizer(self):
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.args.init_learning_rate)

    def set_loss_fn(self):
        if self.args.task_selection == Constants.binary_classification:
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(
                from_logits=False,
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            )

        if self.args.task_selection == Constants.multi_classification:
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
                from_logits=False,
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            )

        if self.args.task_selection == Constants.regression:
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
                from_logits=False,
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            )

    def set_callbacks(self):
        monitor = None
        min_delta = None

        if (self.args.task_selection == Constants.binary_classification or
                self.args.task_selection == Constants.multi_classification):
            monitor = "val_auc"
            # min_delta = 0.01
        elif self.args.task_selection == Constants.regression:
            monitor = "val_loss"
            # min_delta = 0.1
        print("Monitor for Callback:",monitor)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=self.args.early_stop
                                                             , verbose=1, restore_best_weights=True)
        self.callbacks = [early_stopping_cb]

    def set_metrics(self):
        if self.args.task_selection == Constants.binary_classification:
            self.metrics = [tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                            tf.keras.metrics.BinaryAccuracy(),
                            tf.keras.metrics.AUC(curve="PR", name="auc_pr")]

        if self.args.task_selection == Constants.multi_classification:
            self.metrics = [tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.AUC(name="pr"),
                            tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                            tf.keras.metrics.CategoricalAccuracy()]

        if self.args.task_selection == Constants.regression:
            self.metrics = ["mse", "mae"]

    def compile(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.metrics
        )

    def configure_data(self):
        self.data = RNNData(self.args)
        self.train_batch = self.data.train_batch
        self.validation_batch = self.data.validation_batch
        self.test_batch = self.data.test_batch
        print("----- Data -----")
        print("Train Samples Length:", str(self.data.train_df.shape[0]))
        print("Validation Samples Length:", str(self.data.validation_df.shape[0]))
        print("Test Samples Length:", str(self.data.test_df.shape[0]))
        print("Train Batch Length:", str(len(self.train_batch)))
        print("Validation Batch Length:", str(len(self.validation_batch)))
        print("Test Batch Length:", str(len(self.test_batch)))

    def fit(self):
        return self.model.fit(
            self.train_batch,
            validation_data=self.validation_batch,
            epochs=self.args.epochs,
            verbose=1,
            callbacks=self.callbacks,
        )

    def evaluate_only(self):
        self.configure_data()
        self.model = tf.keras.models.load_model(self.args.saved_model_pathway+"/save/")
        evaluation = self.model.evaluate(self.test_batch)

    def run_pipeline(self):
        self.configure_data()
        self.configure_model()
        print("----- Model Summary Before Fit -----")
        print(self.model.summary())
        print("----- Fit Begin -----")
        self.history = self.fit()
        print("----- Fit Complete -----")

        # Save experiment configurations
        if not os.path.exists("./output/" + self.output_filename):
            os.makedirs("./output/" + self.output_filename)
        save_pathway = "./output/" + self.output_filename + "/save/"
        self.model.save(save_pathway)
        vars_dict = vars(self.args)
        config_df = pd.DataFrame(list(vars_dict.items()), columns=['Argument', 'Value'])
        config_df.to_csv("./output/" + self.output_filename + "/config.csv", index=False)

        # Save experiment results
        history = pd.DataFrame(self.history.history)
        evaluation = self.model.evaluate(self.test_batch)
        model_predictions = []
        model_predictions_arr = self.model.predict(self.test_batch)
        for p in model_predictions_arr:
            model_predictions.extend(p)
        # model_predictions = [p[0] for p in self.model.predict(self.test_batch)]
        true_labels = self.data.test_df["label"].to_numpy()
        predictions = pd.DataFrame(data={"predictions": model_predictions, "true_labels": true_labels})
        history.to_csv("./output/" + self.output_filename + "/" + self.output_filename + "_history.csv")
        predictions.to_csv("./output/" + self.output_filename + "/" + self.output_filename + "_predictions.csv")

