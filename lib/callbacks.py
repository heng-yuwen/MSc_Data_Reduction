# -*- coding: utf-8 -*-
"""Self-defined Keras callbacks.

This module provides dedicated Keras callbacks classes to monitor the training process (model.fit() method) and calculate
the real validation loss and accuracy across the whole validation set. Also, the best model is kept while training based
on validation performance.
"""

import numpy as np
import tensorflow as tf


class MonitorAndSaveParameters(tf.keras.callbacks.Callback):
    """Calculate real performance and kept the best model weights based on validation performance"""

    def __init__(self, metrics, batch_size, val_samples=0, early_stop=False):
        """Init a new callback instance.

        :param metrics: a dict object to save processed metrics like loss and accuracy.
        :param batch_size: the size of the mini-batch.
        :param val_samples: number of validation samples.
        """
        super().__init__()

        if isinstance(metrics, dict):
            self.metrics = metrics
        self.train_loss = np.array([])
        self.train_acc = np.array([])
        self.valid_loss = np.array([])
        self.valid_acc = np.array([])
        self.pred_loss = np.array([])
        self.pred_acc = np.array([])
        self.max_acc = 0
        self.best_weights = None
        self.batch_size = batch_size
        self.val_samples = val_samples
        self.residual_samples = val_samples % batch_size
        self.early_stop = early_stop
        if self.residual_samples == 0:
            self.residual_samples = batch_size

    # called for training.
    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if "loss" in keys:
            self.train_loss = np.append(self.train_loss, logs["loss"])
        if "accuracy" in keys:
            self.train_acc = np.append(self.train_acc, logs["accuracy"])
        # print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    # called for validation enabled mode.
    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if "loss" in keys:
            self.valid_loss = np.append(self.valid_loss, logs["loss"])
        if "accuracy" in keys:
            self.valid_acc = np.append(self.valid_acc, logs["accuracy"])
        # print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    # called for model.predict() method.
    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if "loss" in keys:
            self.pred_loss = np.append(self.pred_loss, logs["loss"])
        if "accuracy" in keys:
            self.pred_acc = np.append(self.pred_acc, logs["accuracy"])
        # print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))

    # called at the end of each training epoch
    def on_epoch_end(self, epoch, logs=None):
        # save the model weights if the acc is higher than record.

        if self.metrics["val_acc"][-1] >= self.max_acc:
            self.max_acc = self.metrics["val_acc"][-1]
            if self.early_stop:
                self.best_weights = self.model.get_weights()
        self.valid_loss = np.array([])
        self.valid_acc = np.array([])
        # print("Epoch end")

    # called at the end of the training
    def on_train_end(self, logs=None):
        if self.max_acc > 0 and self.early_stop:
            self.model.set_weights(self.best_weights)
            print("Restoring best model weights with validation accuracy: {}".format(self.max_acc))

    # called at the end of the test/validate process
    def on_test_end(self, logs=None):
        if len(self.valid_loss) > 0:
            if len(self.valid_loss) > 1:
                avg_valid_loss = (self.valid_loss[:-1].sum() * self.batch_size + self.valid_loss[
                    -1] * self.residual_samples) / self.val_samples
                avg_valid_acc = (self.valid_acc[:-1].sum() * self.batch_size + self.valid_acc[
                    -1] * self.residual_samples) / self.val_samples
            else:
                avg_valid_loss = self.valid_loss[0]
                avg_valid_acc = self.valid_acc[0]

            if "val_loss" in self.metrics.keys():
                self.metrics["val_loss"].append(avg_valid_loss)
            else:
                self.metrics["val_loss"] = [avg_valid_loss]
            if "val_acc" in self.metrics.keys():
                self.metrics["val_acc"].append(avg_valid_acc)
            else:
                self.metrics["val_acc"] = [avg_valid_acc]
