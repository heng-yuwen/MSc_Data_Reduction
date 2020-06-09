# -*- coding: utf-8 -*-
"""Feature extractor.

This module provides classes for extracting features from raw image inputs with the pre-trained NASNetLarge model
(called "extractor"). The pre-trained weights on imagenet is provided by TF Hub
(https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization

from .callbacks import MonitorAndSaveParameters


class FeatureExtractor(object):
    """Generic feature extractor."""

    def __init__(self, image_size, classes, hub_url=None, extractor_path=None, trainable=False, input_size=224):
        """Create a  new  feature extractor instance.

        :param image_size: the size of the input image, which should be a square matrix.
        :param classes: the number of classes of the images.
        :param hub_url: the url of the pre-trained extractor from TF Hub.
        :param extractor_path: the path of the saved extractor weights.
        :param trainable: whether to freeze the extractor weights or not.
        :param input_size: the input layer shape of the extractor.
        """

        # build the extractor.
        self.input_size = input_size
        if extractor_path is None:
            self.extractor = tf.keras.Sequential([
                hub.KerasLayer(hub_url,
                               trainable=trainable)
            ])
        else:
            self.extractor = tf.keras.models.load_model(extractor_path)
        self.extractor.build([None, self.input_size, self.input_size, 3])

        # build the full model (add resize image layer to fit the input shape of the extractor)
        self.model_full = None
        self.model_full = tf.keras.Sequential([
            Input(shape=(image_size, image_size, 3)),
            Lambda(lambda image: tf.image.resize(image, [self.input_size, self.input_size])),
            self.extractor
        ])

        # build the compression layer to encode the features a step further.
        self.compressor = tf.keras.Sequential()
        self.compressor.add(
            Dense(128, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.1)))
        self.compressor.add(BatchNormalization())

        # build softmax classification layers.
        self.classifier = tf.keras.Sequential()
        self.classifier.add(self.compressor)
        self.classifier.add(Dense(classes,  # output dim is one score per each class
                                  activation='softmax',
                                  kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.1),
                                  ))

        self.classifier.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

        # build fine-tune model to train all the layers.
        self.fine_tune_model = tf.keras.Sequential([
            self.model_full,
            self.classifier
        ])

        # build the compression model, only be used for fine-tuned network.
        self.compression_model = tf.keras.Sequential([
            self.model_full,
            self.compressor
        ])

        # save the extracted features.
        self.extracted_features = None

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
            self.extracted_features = self.model_full.predict(x, batch_size=batch_size, verbose=1)
        if compression:
            self.extracted_features = self.compression_model.predict(x, batch_size=batch_size, verbose=1)
        return self.extracted_features

    def save_features(self, path):
        """Save the extracted features to the given path.

        :param path: the path of the file, use ".csv" as the file format.
        """
        if not isinstance(self.extracted_features, np.ndarray):
            raise AttributeError("Extracted features not exist, please call .extract() first")
        else:
            pd.DataFrame(self.extracted_features).to_csv(path, index=False)

    def load_features(self, path):
        """Load the saved features.

        :param path: the path of the saved feature file, use them to fine-tune the classifier.
        """
        self.extracted_features = pd.read_csv(path, index_col=False).values

    def test_feature_quality(self, y, epochs=25, batch_size=128, validation_data=None):
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
            print("Extracting features for validation data")
            (x_valid, y_valid) = validation_data
            x_valid = self.extract(x_valid, batch_size)
            validation_data = (x_valid, y_valid)

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
        self.classifier.save_weights(path)

    def load_classifier(self, path):
        """Load the classifier.

        :param path: path string, with format ".h5".
        """
        self.classifier.load_weights(path)

    @property
    def features(self):
        """Get the extracted features"""
        if not isinstance(self.extracted_features, np.ndarray):
            raise AttributeError("Extracted features not exist, please call .extract() first")

        return self.extracted_features


class NASNetLargeExtractor(FeatureExtractor):
    """Build the extractor with pre-trained NASNetLarge model"""

    def __init__(self, image_size, classes):
        """ init a new NASNetLarge extractor instance.

        :param image_size: the size of the input image, which should be a square matrix.
        :param classes: the number of classes of the images.
        """
        super().__init__(image_size, classes,
                         hub_url="https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
                         input_size=331)

    def extract_fine_tuned_features(self, x, y, epochs=25, batch_size=128, validation_data=None):
        """fine-tune the model and extract compressed features.

        :param x: training images.
        :param y: training set labels.
        :param epochs: the number of epochs to train the classifier.
        :param batch_size: the size of the mini-batch.
        :param validation_data: the validation set used to tune the networks.
        """

        # extract features and train the classifier.
        if not self.model_full:
            self.extract(x, batch_size)
        if not self.classifier:
            self.test_feature_quality(y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

        # turn on training
        for layer in self.fine_tune_model.layers:
            layer.trainable = True

        # compile the model (should be done *after* setting layers to non-trainable)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.045,
            decay_steps=1000,
            decay_rate=0.94,
            staircase=True)
        self.fine_tune_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr_schedule, epsilon=1),
                                     loss='categorical_crossentropy',
                                     metrics=['accuracy'])
        history = {}
        default = self.fine_tune_model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data,
                                           callbacks=[
                                               MonitorAndSaveParameters(history, batch_size, len(validation_data[0]))])
        history["default"] = default

        # after fine-tuning, return extracted features
        for layer in self.extractor.layers:
            layer.trainable = False
        features = self.extract(x, batch_size, compression=True)

        return history, features
