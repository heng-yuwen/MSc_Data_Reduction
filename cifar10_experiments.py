'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import numpy as np

from lib.densenet import DenseNet121
from lib.experiments import load_dataset, run_wcl, train_with_original, run_pop, run_egdis, run_cl

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--experiment', default=1, type=int, help='Run which experiment')
parser.add_argument('--select', default=1, type=int, help='Run which stage')
args = parser.parse_args()

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("cifar10")
print("There are {} training samples and {} validation samples".format(x_train.shape[0], x_valid.shape[0]))
print("There are {} test samples.".format(x_test.shape[0]))

batch_size = 128
net = DenseNet121()

# Experiment 1: train whole cifar10  with DenseNet121
if args.experiment == 1:
    history = train_with_original((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10",
                                  batch_size=batch_size)

    np.save(os.path.join(os.getcwd(), "models", "cifar10", "whole_train_his.npy"), history)
    print("History saved.")

# Experiment 2: train the POP selected dataset
if args.experiment == 2:
    history = run_pop((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10,
                      batch_size=batch_size, i=args.select)
    for his in history:
        np.save(os.path.join(os.getcwd(), "models", "cifar10", "pop_his_size_" + str(his["size"]) + ".npy"), history)
    print("History saved.")

# Experiment 3: train the EGDIS selected dataset
if args.experiment == 3:
    history = run_egdis((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10,
                        batch_size=batch_size)
    np.save(os.path.join(os.getcwd(), "models", "cifar10", "egdis_his.npy"), history)
    print("History saved.")

# Experiment 4: train the CL selected dataset
if args.experiment == 4:
    history = run_cl((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10,
                     batch_size=batch_size, i=args.select)
    for his in history:
        np.save(os.path.join(os.getcwd(), "models", "cifar10", "cl_his_size_" + str(his["size"]) + ".npy"), history)
    print("History saved.")

# Experiment 5: train the WCL selected dataset
if args.experiment == 5:
    history = run_wcl((x_train, y_train), (x_valid, y_valid), (x_test, y_test), net, "cifar10", 10,
                      batch_size=batch_size, i=args.select)
    for his in history:
        np.save(os.path.join(os.getcwd(), "models", "cifar10", "wcl_his_size_" + str(his["size"]) + ".npy"), history)
    print("History saved.")
