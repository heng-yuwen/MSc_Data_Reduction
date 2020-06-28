# -*- coding: utf-8 -*-
"""Feature extractor.

This module provides classes for extracting features from raw image inputs with the pre-trained NASNetLarge model
(called "extractor"). The pre-trained weights on imagenet is provided by TF Hub
(https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4).
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Dense, BatchNormalization

from .callbacks import MonitorAndSaveParameters


class FeatureExtractor(object):
    """Generic feature extractor."""

    def __init__(self, image_size, model_path, data_path, hub_url=None, trainable=False,
                 input_size=224, require_resize=True):
        """Create a  new  feature extractor instance.

        :param data_path: the the directory of extracted features.
        :param model_path: the directory of trained model parameters.
        :param image_size: the size of the input image, which should be a square matrix.
        :param hub_url: the url of the pre-trained extractor from TF Hub.
        :param trainable: whether to freeze the extractor weights or not.
        :param input_size: the input layer shape of the extractor.
        """

        self.model_path = model_path
        self.data_path = data_path

        # build the extractor.
        if require_resize:
            self.extractor = tf.keras.Sequential([
                Lambda(lambda image: tf.image.resize(image, [input_size, input_size])),
                hub.KerasLayer(hub_url,
                               trainable=trainable)
            ])
            self.extractor.build([None, image_size, image_size, 3])
        else:
            self.extractor = tf.keras.Sequential([
                hub.KerasLayer(hub_url,
                               trainable=trainable)
            ])
            self.extractor.build([None, input_size, input_size, 3])

        self.compressor_layer = None
        self.classifier_layer = None

        self.compressor = None
        self.classifier = None
        # save the extracted features.
        self.extracted_features = None
        self.extracted_valid_features = None
        self.extracted_compressed_features = None

    def extract(self, x, y=None, batch_size=128, compression=False):
        """Extract the features from training images.

        :param y: label of training data
        :param x: training images.
        :param batch_size: the size of the mini-batch.
        :param compression: compress the features or not, default False, please first train the compressor layer first.
        :return: extracted features.
        """
        features = None
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not compression:
            print("Extracting features...")
            features = self.extractor.predict(x, batch_size=batch_size, verbose=1)
            self.extracted_features = features
        if compression:
            print("Compressing features...")
            features = self.compressor_layer.predict(x, batch_size=batch_size, verbose=1)
            features = np.append(features, y, axis=1)
            self.extracted_compressed_features = features

        return features

    def _extract(self, x, batch_size):
        """Extract features from extractor without compression.

        :param x: any images.
        :param batch_size: the size of the mini-batch.
        :return: extracted features.
        """
        return self.extractor.predict(x, batch_size=batch_size, verbose=1)

    def save_features(self, train=True, valid=True, compressed=True):
        """Save the extracted features to the given path.

        :param compressed: weather to save compressed features or not
        :param valid: weather to save validation features or not
        :param train: weather to save training features or not
        """

        if isinstance(self.extracted_features, np.ndarray) and train:
            pd.DataFrame(self.extracted_features).to_csv(os.path.join(self.data_path, "extracted_train.csv"),
                                                         index=False)
            print("Extracted training set features saved")
        if isinstance(self.extracted_valid_features, np.ndarray) and valid:
            pd.DataFrame(self.extracted_valid_features).to_csv(os.path.join(self.data_path, "extracted_valid.csv"),
                                                               index=False)
            print("Extracted validation set features saved")
        if isinstance(self.extracted_compressed_features, np.ndarray) and compressed:
            pd.DataFrame(self.extracted_compressed_features).to_csv(
                os.path.join(self.data_path, "compressed_train.csv"), index=False)
            print("Compressed training set features saved")

    def load_features(self):
        """Load the saved features."""

        if os.path.exists(os.path.join(self.data_path, "extracted_train.csv")):
            self.extracted_features = pd.read_csv(os.path.join(self.data_path, "extracted_train.csv"),
                                                  index_col=False).values
            print("Extracted training set features loaded")
        if os.path.exists(os.path.join(self.data_path, "extracted_valid.csv")):
            self.extracted_valid_features = pd.read_csv(os.path.join(self.data_path, "extracted_valid.csv"),
                                                        index_col=False).values
            print("Extracted validation set loaded")
        if os.path.exists(os.path.join(self.data_path, "compressed_train.csv")):
            self.extracted_compressed_features = pd.read_csv(os.path.join(self.data_path, "compressed_train.csv"),
                                                             index_col=False).values
            print("Compressed training set features loaded")

    def train_classifier(self, y, epochs=25, batch_size=128, learning_rate=0.01, validation_data=None, early_stop=True):
        """Train the classifier with the extracted features and report performance.

        :param learning_rate: the learning rate used by the SGD.
        :param early_stop: weather return the best model evaluated with validation or not
        :param y: training set labels.
        :param epochs: the number of epochs to train the classifier.
        :param batch_size: the size of the mini-batch.
        :param validation_data: the validation set used to tune the networks.
        """
        if not isinstance(self.extracted_features, np.ndarray):
            raise AttributeError("Extracted features not exist, please call .extract() first")
        # extract features for validation data.
        if validation_data:
            (x_valid, y_valid) = validation_data
            if not isinstance(self.extracted_valid_features, np.ndarray):
                print("Extracting features for validation data")
                self.extracted_valid_features = self._extract(x_valid, batch_size)
            validation_data = (self.extracted_valid_features, y_valid)

        # update learning rate
        K.set_value(self.classifier.optimizer.learning_rate, learning_rate)
        print("The learning rate of the classifier is: {}".format(self.classifier.optimizer.learning_rate.numpy()))

        # train the classifier
        history = self.train(self.classifier, self.extracted_features, y, epochs=epochs, batch_size=batch_size,
                             validation_data=validation_data, early_stop=early_stop)
        # default = self.classifier.fit(self.extracted_features, y, epochs=epochs, batch_size=batch_size,
        #                               validation_data=validation_data, callbacks=[
        #         MonitorAndSaveParameters(history, batch_size, len(validation_data[0]))])
        # history["default"] = default
        return history

    def save_extractor(self):
        """Save the extractor to the given path."""

        self.extractor.save_weights(os.path.join(self.model_path, "extractor.h5"))
        print("Extractor saved")

    def load_extractor(self):
        """Load the extractor."""

        self.extractor.load_weights(os.path.join(self.model_path, "extractor.h5"))
        for layer in self.extractor.layers:
            layer.trainable = False
        print("Extractor loaded")

    def save_classifier(self):
        """Save the classifier."""

        self.compressor_layer.save_weights(os.path.join(self.model_path, "compressor.h5"))
        self.classifier_layer.save_weights(os.path.join(self.model_path, "classifier.h5"))
        print("Classifier saved")

    def load_classifier(self):
        """Load the classifier."""

        # self.compressor_layer.load_weights(os.path.join(self.model_path, "compressor.h5"))
        self.classifier_layer.load_weights(os.path.join(self.model_path, "classifier.h5"))
        print("Classifier loaded")

    def save_history(self, history, name):
        """Save the training history to a file

        :param history: the history dict returned while training models.
        :param name: the name of the file.
        """
        np.save(os.path.join(self.model_path, name + ".npy"), history)
        print("History saved.")

    def train(self, model, x, y, epochs=25, batch_size=128, validation_data=None, early_stop=False):
        """Train a model with given parameters."""
        history = {}
        default = model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data,
                            callbacks=[
                                MonitorAndSaveParameters(history, batch_size, len(validation_data[0]),
                                                         early_stop=early_stop)])
        history["acc"] = default.history["accuracy"]
        history["loss"] = default.history["loss"]
        return history

    def get_cl_score(self, y, batch_size):
        """Use pre-trained network as score function. Cited: Hacohen G, Weinshall D. On the power of curriculum
        learning in training deep networks[J]. arXiv preprint arXiv:1904.03626, 2019. """
        scores = self.classifier_layer.predict_proba(self.extracted_compressed_features[:, :128], batch_size=batch_size,
                                                     verbose=1)
        assert scores.shape == y.shape, "The shapes don't match"
        scores = scores * y
        return scores.sum(axis=1)

    @property
    def features(self):
        """Get the extracted features"""
        if not isinstance(self.extracted_features, np.ndarray):
            raise AttributeError("Extracted features not exist, please call .extract() first")

        return self.extracted_features


class NASNetLargeExtractor(FeatureExtractor):
    """Build the extractor with pre-trained NASNetLarge model"""

    def __init__(self, image_size, classes, model_path, data_path, require_resize=True):
        """ init a new NASNetLarge extractor instance.

        :param image_size: the size of the input image, which should be a square matrix.
        :param classes: the number of classes of the images.
        """
        super().__init__(image_size,
                         # hub_url="https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
                         hub_url=os.path.join(os.getcwd(), "lib", "cache_model", "nasnet"),
                         input_size=331, model_path=model_path, data_path=data_path, require_resize=require_resize)

        # build the compression layer to encode the features a step further.
        self.compressor_layer = tf.keras.Sequential([
            Dense(128, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.1)),
            BatchNormalization()
        ])
        self.compressor_layer.build([None, 4032])

        # build softmax classification layers.
        self.classifier_layer = tf.keras.Sequential([
            Dense(classes,  # output dim is one score per each class
                  activation='softmax',
                  kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.1),
                  )
        ])
        self.classifier_layer.build([None, 128])

        # build the full model (add resize image layer to fit the input shape of the extractor)
        self.model = tf.keras.Sequential([
            self.extractor,
            self.compressor_layer,
            self.classifier_layer
        ])

        # build softmax classification layers.
        self.classifier = tf.keras.Sequential([
            self.compressor_layer,
            self.classifier_layer
        ])
        self.classifier.compile(optimizer=tf.keras.optimizers.SGD(0.01),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

        # build the compression model, only be used for fine-tuned network.
        self.compressor = tf.keras.Sequential([
            self.extractor,
            self.compressor_layer,
        ])

    def fine_tune_features(self, x, y, learning_rate=0.0001, weight_decay=0.01, epochs=25, batch_size=128,
                           validation_data=None, early_stop=False):
        """fine-tune the model and extract compressed features.

        :param weight_decay: the l2 regularization weight decay.
        :param learning_rate: the learning rate of SGD.
        :param x: training images.
        :param y: training set labels.
        :param epochs: the number of epochs to train the classifier.
        :param batch_size: the size of the mini-batch.
        :param validation_data: the validation set used to tune the networks.
        """

        # load trained parameters
        self.load_classifier()
        self.load_extractor()

        # # load extracted features
        # self.load_features()

        # l2 regularisation.
        regularizer = tf.keras.regularizers.l2(weight_decay)

        # turn on training
        for layer in self.model.layers:
            layer.trainable = True
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        history = self.train(self.model, x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data,
                             early_stop=early_stop)
        # default = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data,
        #                          callbacks=[
        #                              MonitorAndSaveParameters(history, batch_size, len(validation_data[0]))])
        # history["default"] = default

        # after fine-tuning, return extracted features
        for layer in self.extractor.layers:
            layer.trainable = False
        # features = self.extract(x, batch_size, compression=True)

        return history