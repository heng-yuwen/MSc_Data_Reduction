'''Train CIFAR10 with PyTorch.'''
import os

import numpy as np

from lib.densenet import DenseNet121
from lib.experiments import load_dataset, run_pop

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("cifar10")
print("There are {} training samples and {} validation samples".format(x_train.shape[0], x_valid.shape[0]))
print("There are {} test samples.".format(x_test.shape[0]))

batch_size = 256
net = DenseNet121()

# Experiment 1: train whole cifar10  with DenseNet121
# history = train_with_original((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10",
#                               batch_size=batch_size)
#
# np.save(os.path.join(os.getcwd(), "models", "cifar10", "whole_train_his.npy"), history)
# print("History saved.")

# Experiment 2: train the POP selected dataset
history = run_pop((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10, batch_size=batch_size)
for his in history:
    np.save(os.path.join(os.getcwd(), "models", "cifar10", "pop_his_size_" + str(his["size"]) + ".npy"), history)
print("History saved.")

# Experiment 3: train the EGDIS selected dataset
# history = run_egdis((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10, batch_size=batch_size)
# np.save(os.path.join(os.getcwd(), "models", "cifar10", "egdis_his.npy"), history)
# print("History saved.")

# Experiment 4: train the CL selected dataset
# history = run_cl((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10, batch_size=batch_size)
# for his in history:
#     np.save(os.path.join(os.getcwd(), "models", "cifar10", "cl_his_size_" + str(his["size"]) + ".npy"), history)
# print("History saved.")

# Experiment 5: train the WCL selected dataset
# history = run_wcl((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10, batch_size=batch_size)
# for his in history:
#     np.save(os.path.join(os.getcwd(), "models", "cifar10", "wcl_his_size_" + str(his["size"]) + ".npy"), history)
# print("History saved.")
