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
from tensorflow.keras.layers import Lambda, Dense, BatchNormalization, Input

from .callbacks import MonitorAndSaveParameters


class FeatureExtractor(object):
    """Generic feature extractor."""

    def __init__(self, image_size, classes, hub_url=None, trainable=False, input_size=224,
                 data_name="default"):
        """Create a  new  feature extractor instance.

        :param image_size: the size of the input image, which should be a square matrix.
        :param classes: the number of classes of the images.
        :param hub_url: the url of the pre-trained extractor from TF Hub.
        :param trainable: whether to freeze the extractor weights or not.
        :param input_size: the input layer shape of the extractor.
        :param data_name: the name of the dataset, used as the dataset folder name.
        """

        self.data_name = data_name

        # build the whole model.
        self.model = tf.keras.Sequential([
            Input(shape=(image_size, image_size, 3)),
            Lambda(lambda image: tf.image.resize(image, [input_size, input_size])),
            hub.KerasLayer(hub_url,
                           trainable=trainable, name="extractor"),
            Dense(128, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.1), name="classifier"),
            BatchNormalization(name="compressor"),
            Dense(classes,  # output dim is one score per each class
                  activation='softmax',
                  kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.1),
                  )
        ])
        # self.model.build([None, image_size, image_size, 3])

        # build the extractor
        self.extractor = tf.keras.Sequential([
            self.model.input,
            self.model.get_layer("extractor")
        ])

        # build the classifier
        self.classifier = tf.keras.Sequential([
            self.model.get_layer("classifier"),
            self.model.output
        ])
        self.classifier.compile(optimizer="sgd",
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

        # build the compression model, only be used for fine-tuned network.
        self.compressor = tf.keras.Sequential([
            self.model.input,
            self.model.get_layer("compressor")
        ])

        # save the extracted features.
        self.extracted_features = None
        self.extracted_valid_features = None
        self.extracted_compressed_features = None

    def extract(self, x, batch_size=128, compression=False):
        """Extract the features from training images.

        :param x: training images.
        :param batch_size: the size of the mini-batch.
        :param compression: compress the features or not, default False, please first train the compressor layer first.
        :return: extracted features.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not compression:
            self.extracted_features = self.extractor.predict(x, batch_size=batch_size, verbose=1)
        if compression:
            self.extracted_compressed_features = self.compressor.predict(x, batch_size=batch_size, verbose=1)
        return self.extracted_features

    def _extract(self, x, batch_size):
        """Extract features from extractor without compression.

        :param x: any images.
        :param batch_size: the size of the mini-batch.
        :return: extracted features.
        """
        return self.extractor.predict(x, batch_size=batch_size, verbose=1)

    def save_features(self, path):
        """Save the extracted features to the given path.

        :param path: the path of the file, use ".csv" as the file format.
        """
        if not isinstance(self.extracted_features, np.ndarray):
            raise AttributeError("Extracted features not exist, please call .extract() first")
        if not isinstance(self.extracted_valid_features, np.ndarray):
            raise AttributeError("Extracted validation features not exist, please call .test_feature_quality() first")

        # save extracted features.
        pd.DataFrame(self.extracted_features).to_csv(os.path.join(path, "train_features_not_compressed.csv"),
                                                     index=False)
        pd.DataFrame(self.extracted_valid_features).to_csv(os.path.join(path, "val_features_not_compressed.csv"),
                                                           index=False)

    def load_features(self, path):
        """Load the saved features.

        :param path: the path of the saved feature file, use them to fine-tune the classifier.
        """
        self.extracted_features = pd.read_csv(path, index_col=False).values

    def train_classifier(self, y, epochs=25, batch_size=128, validation_data=None):
        """Train the classifier with the extracted features and report performance.

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

        # train the classifier
        history = {}
        default = self.classifier.fit(self.extracted_features, y, epochs=epochs, batch_size=batch_size,
                                      validation_data=validation_data, callbacks=[
                MonitorAndSaveParameters(history, batch_size, len(validation_data[0]))])
        history["default"] = default
        return history

    def save_extractor(self, path):
        """Save the extractor to the given path.

        :param path: path string, with format ".h5".
        """
        self.extractor.save_weights(path)

    def load_extractor(self, path):
        """Load the extractor.

        :param path: path string, with format ".h5".
        """
        self.extractor.load_weights(path)
        for layer in self.extractor.layers:
            layer.trainable = False

    def save_classifier(self, path):
        """Save the classifier.

        :param path: path string, with format ".h5".
        :return:
        """
        self.classifier.save_weights(os.path.join(path, "classifier.h5"))

    def load_classifier(self, path):
        """Load the classifier.

        :param path: path string, with format ".h5".
        """
        self.classifier.load_weights(path)

    def save_uncompressed(self):
        """Save the features and extracted features"""

        # create data folder.
        if not os.path.exists(self.data_name):
            os.mkdir(self.data_name)
            print("Directory ", self.data_name, " Created ")

        print("Saving model parameters and extracted features to directory: ", self.data_name)

        # save extracted features.
        self.save_features(self.data_name)
        print("Uncompressed features saved successfully.")

        # save model parameters
        self.save_classifier(self.data_name)

    @property
    def features(self):
        """Get the extracted features"""
        if not isinstance(self.extracted_features, np.ndarray):
            raise AttributeError("Extracted features not exist, please call .extract() first")

        return self.extracted_features


class NASNetLargeExtractor(FeatureExtractor):
    """Build the extractor with pre-trained NASNetLarge model"""

    def __init__(self, image_size, classes, data_name="default"):
        """ init a new NASNetLarge extractor instance.

        :param image_size: the size of the input image, which should be a square matrix.
        :param classes: the number of classes of the images.
        """
        super().__init__(image_size, classes,
                         hub_url="https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
                         input_size=331, data_name=data_name)

    def extract_fine_tuned_features(self, x, y, epochs=25, batch_size=128, validation_data=None):
        """fine-tune the model and extract compressed features.

        :param x: training images.
        :param y: training set labels.
        :param epochs: the number of epochs to train the classifier.
        :param batch_size: the size of the mini-batch.
        :param validation_data: the validation set used to tune the networks.
        """

        # extract features and train the classifier.
        if not isinstance(self.extracted_features, np.ndarray):
            self.extract(x, batch_size)
        if not self.classifier:
            self.train_classifier(y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

        # turn on training
        for layer in self.extractor.layers:
            layer.trainable = True

        # compile the model (should be done *after* setting layers to non-trainable)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.045,
            decay_steps=1000,
            decay_rate=0.94,
            staircase=True)
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr_schedule, epsilon=1),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        history = {}
        default = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data,
                                 callbacks=[
                                     MonitorAndSaveParameters(history, batch_size, len(validation_data[0]))])
        history["default"] = default

        # after fine-tuning, return extracted features
        for layer in self.extractor.layers:
            layer.trainable = False
        features = self.extract(x, batch_size, compression=True)

        return history, features
