# -*- coding: utf-8 -*-
"""Feature extractor.

This module provides classes for extracting features from raw image inputs with the pre-trained NASNetLarge model. The
pre-trained weights on imagenet is provided by TF Hub (https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4).
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Lambda, Input, Dense

from .callbacks import MonitorAndSaveParameters


class FeatureExtractor(object):
    """Generic feature extractor."""

    def __init__(self, image_size, classes, hub_url=None, extractor_path=None, trainable=False, input_size=224):
        self.model_full = None
        self.input_size = input_size
        if extractor_path is None:
            self.hub_model = tf.keras.Sequential([
                hub.KerasLayer(hub_url,
                               trainable=trainable)
            ])
        else:
            self.hub_model = tf.keras.models.load_model(extractor_path)
        self.hub_model.build([None, self.input_size, self.input_size, 3])

        # build the full model
        self.model_full = tf.keras.Sequential([
            Input(shape=(image_size, image_size, 3)),
            Lambda(lambda image: tf.image.resize(image, [self.input_size, self.input_size])),
            self.hub_model
        ])

        # build softmax classification layer.
        self.logistic_model = tf.keras.Sequential()
        self.logistic_model.add(
            Dense(128, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.1), name="feature_layer"))
        self.logistic_model.add(Dense(classes,  # output dim is one score per each class
                                      activation='softmax',
                                      kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.1),
                                      ))  # input dimension = number of features your
        # data has;

        self.logistic_model.compile(optimizer='sgd',
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])

        # build fine-tune model
        # fine-tune the network
        self.fine_tune_model = tf.keras.Sequential([
            self.model_full,
            self.logistic_model
        ])

        # build the compression model, only be used for fine-tuned network.
        self.compression_model = tf.keras.Sequential([
            self.model_full,
            self.logistic_model.get_layer("feature_layer")
        ])

        self.extracted_features = None

    def extract(self, x, batch_size=128, compression=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not compression:
            self.extracted_features = self.model_full.predict(x, batch_size=batch_size, verbose=1)
        if compression:
            self.extracted_features = self.compression_model.predict(x, batch_size=batch_size, verbose=1)
        return self.extracted_features

    def save_features(self, path):
        if not isinstance(self.extracted_features, np.ndarray):
            raise AttributeError("Extracted features not exist, please call .extract() first")
        else:
            pd.DataFrame(self.extracted_features).to_csv(path, index=False)

    def load_features(self, path):
        self.extracted_features = pd.read_csv(path, index_col=False).values

    def test_feature_quality(self, y, epochs=25, batch_size=128, validation_data=None):
        # if type(self.extracted_features) != np.ndarray:
        #     AttributeError("Extracted features not exist, please call .extract() first")
        if not isinstance(self.extracted_features, np.ndarray):
            raise AttributeError("Extracted features not exist, please call .extract() first")

        if validation_data:
            (x_test, y_test) = validation_data
            x_test = self.extract(x_test)
            validation_data = (x_test, y_test)

        # train the dense layer
        history = {}
        default = self.logistic_model.fit(self.extracted_features, y, epochs=epochs, batch_size=batch_size,
                                          validation_data=validation_data, callbacks=[
                MonitorAndSaveParameters(history, batch_size, len(validation_data[0]))])
        history["default"] = default
        return history

    def save_extractor(self, path):
        self.hub_model.save_weights(path)

    def load_extractor(self, path):
        self.hub_model.load_weights(path)
        for layer in self.hub_model.layers:
            layer.trainable = False

    def save_classification_layer(self, path):
        self.logistic_model.save_weights(path)

    def load_classification_layer(self, path):
        self.logistic_model.load_weights(path)

    @property
    def features(self):
        if not isinstance(self.extracted_features, np.ndarray):
            raise AttributeError("Extracted features not exist, please call .extract() first")

        return self.extracted_features


class NASNetLargeExtractor(FeatureExtractor):

    def __init__(self, image_size, classes):
        super().__init__(image_size, classes,
                         hub_url="https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
                         input_size=331)

    def extract_fine_tuned_features(self, x, y, epochs=25, batch_size=128, validation_data=None):
        if not self.model_full:
            self.extract(x, batch_size)
        if not self.logistic_model:
            self.test_feature_quality(y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

        # turn on training
        for layer in self.fine_tune_model.layers:
            layer.trainable = True

        # compile the model (should be done *after* setting layers to non-trainable)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.045,
            decay_steps=10000,
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
        for layer in self.hub_model.layers:
            layer.trainable = False
        features = self.extract(x, batch_size, compression=True)

        return history, features
