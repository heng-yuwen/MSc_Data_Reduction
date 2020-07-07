'''Train CIFAR10 with PyTorch.'''
import os

import numpy as np

from lib.densenet import DenseNet121
from lib.experiments import load_dataset, train_with_original

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("cifar10")
print("There are {} training samples and {} validation samples".format(x_train.shape[0], x_valid.shape[0]))
print("There are {} test samples.".format(x_test.shape[0]))

batch_size = 128
net = DenseNet121()

# Experiment 1: train whole cifar10  with DenseNet121
history = train_with_original((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10")
print(history)

np.save(os.path.join(os.getcwd(), "models", "cifar10", "whole_train_his.npy"), history)
print("History saved.")
