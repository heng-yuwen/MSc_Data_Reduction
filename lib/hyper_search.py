# -*- coding: utf-8 -*-
"""Grid search class for fine-tuning pre-trained keras model.

This model provides methods to search for optimal learning rate and weight decay rate (L2 norm). The search method is
described in the paper "Do Better ImageNet Models Transfer Better?"
"""

import numpy as np


class RandomSearch(object):
    """Hyper-parameter search methods"""

    def __init__(self, model):
        """Create a new GridSearch instance.

        :param model: the pre-trained keras model.
        """

        self.learning_rate = np.power(10, -3 * np.random.rand(7, ) - 1)
        self.weight_decay = np.power(10, -3 * np.random.rand(7, ) - 3)
        self.model = model
        # self.weights = None

    def __call__(self, x, y, validation_data, batch_size):
        """Do the grid search evaluation

        :param x: the training set images.
        :param y: the ground truth labels.
        :param validation_data: validation images and labels.
        :return best_dict, history_dict.
        """

        loss_min = np.inf
        # acc_max = 0
        best_dict = {"learning_rate": None, "weight_decay": None, "acc": 0, "loss": np.inf}
        history_dict = {}
        for lr in self.learning_rate:
            for wd in self.weight_decay:
                print("Evaluating hyper parameters: learning_rate = {}, weight_decay = {}".format(lr, wd))
                history = self.model.fine_tune_features(x, y, learning_rate=lr, weight_decay=wd, epochs=10,
                                                        batch_size=batch_size, validation_data=validation_data)

                # record history
                history_dict[len(history_dict)] = {"learning_rate": lr, "weight_decay": wd, "history": history}

                # check if the minimum loss is smaller than records
                index = np.array(history["val_loss"]).argmin()
                acc_temp = history["val_acc"][index]
                loss_temp = history["val_loss"][index]

                # update best information
                if loss_temp < loss_min:
                    # acc_max = acc_temp
                    loss_min = loss_temp
                    best_dict["learning_rate"] = lr
                    best_dict["weight_decay"] = wd
                    best_dict["acc"] = acc_temp
                    best_dict["loss"] = loss_temp

        print("The best combination is: lr = {}, wd = {}".format(best_dict["learning_rate"], best_dict["weight_decay"]))

        return best_dict, history_dict
