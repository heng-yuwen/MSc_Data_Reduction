# -*- coding: utf-8 -*-
"""Feature extractor.

This module provides classes for extracting features from raw image inputs with the pre-trained NASNetLarge model. The
pre-trained weights on imagenet is provided by TF Hub (https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4).
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Lambda, Input


class FeatureExtractor(object):
    """Generic feature extractor."""

    def __init__(self, hub_url=None, extractor_path=None, trainable=False, input_size=224):
        self.logistic_model = None
        self.model_full = None
        self.input_size = input_size
        if extractor_path is None:
            self.hub_model = tf.keras.Sequential([
                hub.KerasLayer(hub_url,
                               trainable=trainable),
            ])
        else:
            self.hub_model = tf.keras.models.load_model(extractor_path)
        self.hub_model.build([None, self.input_size, self.input_size, 3])
        self.extracted_features = None

    def extract(self, x, batch_size=128):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        image_size = x.shape[1]
        self.model_full = tf.keras.Sequential([
            Input(shape=(image_size, image_size, 3)),
            Lambda(lambda image: tf.image.resize(image, [self.input_size, self.input_size])),
            self.hub_model
        ])
        self.extracted_features = self._extract(x, batch_size=batch_size)
        return self.extracted_features

    def _extract(self, x, batch_size=128):

        return self.model_full.predict(x, batch_size=batch_size, verbose=1)

    def save_features(self, features, path):
        pd.DataFrame(features).to_csv(path, index=False)

    def load_features(self, path):
        return pd.read_csv(path, index_col=False).values

    def test_feature_quality(self, y, epochs=25, batch_size=128, validation_data=None):
        if type(self.extracted_features) != np.ndarray:
            AttributeError("Extracted features not exist, please call .extract() first")

        try:
            (x_test, y_test) = validation_data
            x_test = self._extract(x_test)
            validation_data = (x_test, y_test)
        except:
            validation_data = None

        input_dim = self.extracted_features.shape[1]

        # train the dense layer
        self.logistic_model = tf.keras.Sequential()
        self.logistic_model.add(tf.keras.layers.Dense(10,  # output dim is 10, one score per each class
                                                      activation='softmax',
                                                      kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.1),
                                                      input_dim=input_dim))  # input dimension = number of features your
        # data has

        self.logistic_model.compile(optimizer='sgd',
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])
        self.logistic_model.fit(self.extracted_features, y, epochs=epochs, batch_size=batch_size,
                                validation_data=validation_data)

    def save_extractor(self, path):
        self.hub_model.save_weights(path)

    def load_extractor(self, path):
        self.hub_model.load_weights(path)
        for layer in self.hub_model.layers:
            layer.trainable = False


class NASNetLargeExtractor(FeatureExtractor):

    def __init__(self):
        super().__init__(
            hub_url="https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
            input_size=331)

    def extract_fine_tuned_features(self, x, y, epochs=25, batch_size=128, validation_data=None):
        if not self.model_full:
            self.extract(x, batch_size)
        if not self.logistic_model:
            self.test_feature_quality(y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

        # fine-tune the network
        fine_tune_model = tf.keras.Sequential([
            self.model_full,
            self.logistic_model
        ])

        # turn on training
        for layer in fine_tune_model.layers:
            layer.trainable = True

        # compile the model (should be done *after* setting layers to non-trainable)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.045,
            decay_steps=10000,
            decay_rate=0.94,
            staircase=True)
        fine_tune_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr_schedule, epsilon=1),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
        fine_tune_model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

        # after fine-tunning, return extracted features
        for layer in self.hub_model.layers:
            layer.trainable = False
        return self.extract(x, batch_size)
