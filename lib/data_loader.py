# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CIFAR10 small images classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch


def load_cifar10():
    """Loads [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
    This is a dataset of 50,000 32x32 color training images and 10,000 test
    images, labeled over 10 categories. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        **x_train, x_test**: uint8 arrays of RGB image data with shape
          (num_samples, 3, 32, 32) if the `tf.keras.backend.image_data_format` is
          'channels_first', or (num_samples, 32, 32, 3) if the data format
          is 'channels_last'.
        **y_train, y_test**: uint8 arrays of category labels
          (integers in range 0-9) each with shape (num_samples, 1).
    """
    dirname = 'cifar-10-batches-py'
    # origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = os.path.join(os.getcwd(), "datasets", "cifar10", dirname)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

def load_cifar100(label_mode='fine'):
    """Loads [CIFAR100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
    This is a dataset of 50,000 32x32 color training images and
    10,000 test images, labeled over 100 fine-grained classes that are
    grouped into 20 coarse-grained classes. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).
    Arguments:
        label_mode: one of "fine", "coarse". If it is "fine" the category labels
        are the fine-grained labels, if it is "coarse" the output labels are the
        coarse-grained superclasses.
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        **x_train, x_test**: uint8 arrays of RGB image data with shape
          (num_samples, 3, 32, 32) if the `tf.keras.backend.image_data_format` is
          'channels_first', or (num_samples, 32, 32, 3) if the data format
          is 'channels_last'.
        **y_train, y_test**: uint8 arrays of category labels with shape
          (num_samples, 1).
    Raises:
        ValueError: in case of invalid `label_mode`.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    dirname = 'cifar-100-python'
    # origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = os.path.join(os.getcwd(), "datasets", "cifar100", dirname)

    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    """Loads the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
    This is a dataset of 60,000 28x28 grayscale images of the 10 digits,
    along with a test set of 10,000 images.
    More info can be found at the
    (MNIST homepage)[http://yann.lecun.com/exdb/mnist/].
    Arguments:
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        **x_train, x_test**: uint8 arrays of grayscale image data with shapes
          (num_samples, 28, 28).
        **y_train, y_test**: uint8 arrays of digit labels (integers in range 0-9)
          with shapes (num_samples,).
    License:
        Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
        which is a derivative work from original NIST datasets.
        MNIST dataset is made available under the terms of the
        [Creative Commons Attribution-Share Alike 3.0 license.](
        https://creativecommons.org/licenses/by-sa/3.0/)
    """
    # origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    path = os.path.join(os.getcwd(), "datasets", "mnist", 'mnist.npz')

    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)


def normalize_img(img):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.cast(img, tf.float32)
    height = image.shape[0]
    width = image.shape[1]

    if height == width:
        image = tf.image.resize(image, [331, 331])

    elif height > width:
        ratio = 331. / width
        new_height = int(height * ratio)
        image = tf.image.resize(image, [new_height, 331])
        image = tf.image.crop_to_bounding_box(image, (new_height - 331) // 2, 0, 331, 331)

    else:
        ratio = 331. / height
        new_width = int(width * ratio)
        image = tf.image.resize(image, [331, new_width])
        image = tf.image.crop_to_bounding_box(image, 0, (new_width - 331) // 2, 331, 331)

    image = image / 255.
    return image


def load_dtd():
    path = os.path.join(os.getcwd(), "datasets", "dtd")
    x = []

    # load train data list
    for set_name in ["train1", "val1", "test1"]:
        x_sub = [l for l in open(os.path.join(path, "labels", set_name + ".txt")).readlines()]
        x.extend(x_sub)

    x = np.array(x)
    rng = np.random.default_rng(42)
    rng.shuffle(x)
    x_train_path, x_valid_path, x_test_path = np.split(x, [int(.6 * len(x)), int(.8 * len(x))])

    y_train = [label.split("/")[0] for label in x_train_path]
    y_valid = [label.split("/")[0] for label in x_valid_path]
    y_test = [label.split("/")[0] for label in x_test_path]

    x_train = [normalize_img(np.asarray(Image.open(os.path.join(path, "images", img_path.strip())))) for img_path in
               x_train_path]
    x_valid = [normalize_img(np.asarray(Image.open(os.path.join(path, "images", img_path.strip())))) for img_path in
               x_valid_path]
    x_test = [normalize_img(np.asarray(Image.open(os.path.join(path, "images", img_path.strip())))) for img_path in
              x_test_path]

    return x_train, y_train, x_valid, y_valid, x_test, y_test
