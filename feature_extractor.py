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

    def __init__(self, hub_url, trainable=False, input_size=224):
        self.input_size = input_size
        self.hub_model = tf.keras.Sequential([
            hub.KerasLayer(hub_url,
                           trainable=trainable),
        ])
        self.hub_model.build([None, self.input_size, self.input_size, 3])
        self.extracted_features = None

    def extract(self, x, batch_size):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        image_size = x.shape[1]
        model_full = tf.keras.Sequential([
            Input(shape=(image_size, image_size, 3)),
            Lambda(lambda image: tf.image.resize(image, [self.input_size, self.input_size])),
            self.hub_model
        ])

        self.extracted_features = model_full.predict(x, batch_size=batch_size, verbose=1)

    @property
    def features(self):
        return self.extracted_features

    def save_features(self, path):
        pd.DataFrame(self.features).to_csv(path)


class NASNetLargeExtractor(FeatureExtractor):

    def __init__(self):
        super(FeatureExtractor, self).__init__(
            hub_url="https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
            input_size=331)
