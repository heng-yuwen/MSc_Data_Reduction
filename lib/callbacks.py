import numpy as np
import tensorflow as tf


class MonitorAndSaveParameters(tf.keras.callbacks.Callback):
    def __init__(self, metrics, batch_size, val_samples):
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
        if len(self.valid_loss) > 0:
            if len(self.valid_loss) > 1:
                avg_valid_loss = (self.valid_loss[:-1].sum() * self.batch_size + self.valid_loss[
                    -1] * self.residual_samples) / self.val_samples
                avg_valid_acc = (self.valid_acc[:-1].sum() * self.batch_size + self.valid_acc[
                    -1] * self.residual_samples) / self.val_samples
            else:
                avg_valid_loss = self.valid_loss[0]
                avg_valid_acc = self.valid_acc[0]
            # print(self.valid_loss)
            # print(self.valid_acc)
            # print("Epoch: {} finished. The average validation loss is: {}. The validation accuracy is: {}".format(epoch,
            #                                                                                                       avg_valid_loss,
            #                                                                                                       avg_valid_acc))
            if "val_loss" in self.metrics.keys():
                self.metrics["val_loss"].append(avg_valid_loss)
            else:
                self.metrics["val_loss"] = [avg_valid_loss]

            if "val_acc" in self.metrics.keys():
                self.metrics["val_acc"].append(avg_valid_acc)
            else:
                self.metrics["val_acc"] = [avg_valid_acc]

            # save the model weights if the acc is higher than record.
            if avg_valid_acc >= self.max_acc:
                self.max_acc = avg_valid_acc
                self.best_weights = self.model.get_weights()
            self.valid_loss = np.array([])
            self.valid_acc = np.array([])

    # called at the end of the training
    def on_train_end(self, logs=None):
        if self.max_acc > 0:
            self.model.set_weights(self.best_weights)
            print("Restoring best model weights with validation accuracy: {}".format(self.max_acc))
