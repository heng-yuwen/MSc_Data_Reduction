'''Train CIFAR10 with PyTorch.'''
import os
from functools import reduce

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset

from lib.data_loader import load_cifar10, load_cifar100
from lib.reduction_algorithms import POP
from .utils import progress_bar


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, ndarrays, transform=None):
        assert all(ndarrays[0].shape[0] == array.shape[0] for array in ndarrays)
        self.ndarrays = ndarrays
        self.transform = transform

    def __getitem__(self, index):
        x = self.ndarrays[0][index]

        if self.transform:
            x = self.transform(x)

        y = torch.tensor(self.ndarrays[1][index], dtype=torch.long)

        return x, y

    def __len__(self):
        return self.ndarrays[0].shape[0]


def load_dataset(dataset):
    if dataset == "cifar10":
        # load cifar10 datasets
        (x_train, y_train), (x_test, y_test) = load_cifar10()
    elif dataset == "cifar100":
        # load cifar10 datasets
        (x_train, y_train), (x_test, y_test) = load_cifar100()
    else:
        raise AttributeError("Dataset name \"{}\" is not supported.".format(dataset))

    y_train = y_train.reshape(-1, )
    y_test = y_test.reshape(-1, )
    # split a validation set
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def load_compressed_train_set(dataset, classes):
    data = pd.read_csv(os.path.join("datasets", dataset, "compressed_train.csv")).values
    data_x = data[:, :-classes]
    data_y = data[:, -classes:]
    data_y = np.argmax(data_y, axis=1)

    return data_x, data_y


# Training
def train_epoch(trainloader, epoch, net, device, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader),
                     "Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})".format(train_loss / total, 100. * correct / total,
                                                                      correct, total))
    acc = 100. * correct / total
    err = train_loss / total
    return acc, err


def valid_epoch(validloader, epoch, net, device, criterion, best_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validloader),
                         "Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})".format(test_loss / total, 100. * correct / total,
                                                                          correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    err = test_loss / total
    if acc > best_acc:
        print('Saving with better validation accuracy model: {:.3f}..'.format(acc))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        best_acc = acc
    else:
        state = None

    return acc, err, best_acc, state


def test_epoch(testloader, net, device, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader),
                         "Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})".format(test_loss / total,
                                                                          100. * correct / total, correct, total))

    acc = 100. * correct / total
    err = test_loss / total
    return acc, err


def check_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def train_with_original(train, valid, test, net, dataset, batch_size=128, name="whole", stage=1):
    best_acc = 0  # best test accuracy
    best_state = None

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    device = check_device()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        print("Use multiple GPU to train the network.")

    criterion = nn.CrossEntropyLoss()

    trainset = CustomTensorDataset(ndarrays=(train[0], train[1]), transform=transform_train)
    validset = CustomTensorDataset(ndarrays=(valid[0], valid[1]), transform=transform_test)
    testset = CustomTensorDataset(ndarrays=(test[0], test[1]), transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    history = {"acc": [], "val_acc": [], "loss": [], "val_loss": [], "test_acc": 0, "test_loss": 0}

    if stage == 1:
        optimizer = optim.SGD(net.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        epochs = 150
    elif stage == 2:

        optimizer = optim.SGD(net.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=5e-4)
        epochs = 100
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.001,
                              momentum=0.9, weight_decay=5e-4)
        epochs = 100

    print("Use {} epochs".format(epochs))

    for epoch in range(epochs):
        train_acc, train_loss = train_epoch(trainloader, epoch, net, device, optimizer, criterion)
        valid_acc, valid_loss, best_acc, state = valid_epoch(validloader, epoch, net, device, criterion, best_acc)
        if state:
            best_state = state

        history["acc"].append(train_acc)
        history["loss"].append(train_loss)
        history["val_acc"].append(valid_acc)
        history["val_loss"].append(valid_loss)

    test_acc, test_loss = test_epoch(testloader, net, device, criterion)
    history["test_acc"] = test_acc
    history["test_loss"] = test_loss

    print("Saving best model parameters with validation acc: {}".format(best_acc))
    torch.save(best_state,
               os.path.join(os.getcwd(), "models", dataset, name + "_stage_" + str(stage) + "_set_ckpt.pth"))

    return history


def run_pop(train, valid, test, net, dataset, classes, batch_size=128, i=1, stage=1, num_samples=0):
    compressed_train_x, compressed_train_y = load_compressed_train_set(dataset, classes)
    pop = POP()

    print("Now try to run the algorithm POP")

    sample_weakness = pop.fit(compressed_train_x, compressed_train_y)
    print("------------------ Start to select subsets ------------------")
    history = []

    if num_samples != 0:
        print("Start to select {} samples for a fair comparison.".format(num_samples))
        i = 1
        while len(np.argwhere(sample_weakness <= i)) <= num_samples:
            i += 1

        i -= 1
        selected_idx = np.argwhere(sample_weakness <= i)
        print("Selected {} samples for weakness < {}".format(len(selected_idx), i))
        num_diff = num_samples - len(selected_idx)
        if num_diff != 0:
            print("Now select extra {} samples by random".format(num_diff))
            next_weakness_idx = np.setdiff1d(np.argwhere(sample_weakness <= i + 1), np.argwhere(sample_weakness <= i))
            diff_idx = next_weakness_idx[np.random.choice(len(next_weakness_idx), num_diff, replace=False)]
            subset_idx = np.union1d(selected_idx, diff_idx)
        else:
            subset_idx = selected_idx
        size = len(subset_idx)

    else:
        # for i in range(1, int(sample_weakness.max()), 3):
        subset_idx = np.argwhere(sample_weakness <= i).reshape(-1)
        size = len(subset_idx)
        print("Selected {} samples for weakkness <= {}.".format(size, i))

    print("Train with the selected {} samples.".format(len(subset_idx)))
    his = train_with_original((train[0][subset_idx], train[1][subset_idx]), valid, test, net, dataset,
                              batch_size=batch_size, name="pop", stage=stage)
    his["size"] = size

    history.append(his)
    print("------------------------------------------------------------")

    return history


def run_egdis(train, valid, test, net, dataset, classes, batch_size=128, stage=1):
    compressed_train_x, compressed_train_y = load_compressed_train_set(dataset, classes)
    # egdis = EGDIS()

    print("Now try to run the algorithm EGDIS with the generated sample dataset.")
    selected_egdis_idx = np.load(os.path.join(os.getcwd(), "datasets", dataset, "selected_egdis_idx.npy"))

    print("Selected {} samples".format(len(selected_egdis_idx)))

    history = train_with_original((train[0][selected_egdis_idx], train[1][selected_egdis_idx]), valid, test, net,
                                  dataset, batch_size=batch_size, name="egdis", stage=stage)
    history["size"] = len(selected_egdis_idx)
    return history


def run_cl(train, valid, test, net, dataset, classes, batch_size=128, i=1, stage=1, num_samples=0):
    compressed_train_x, compressed_train_y = load_compressed_train_set(dataset, classes)
    # cl = CL()
    # cl.fit_dataset(classes=classes, dataset=dataset)
    # rank, scores = cl.fit(compressed_train_x, to_categorical(compressed_train_y, num_classes=classes))
    scores = np.load(os.path.join(os.getcwd(), "datasets", dataset, "cl_scores.npy"))
    history = []
    print("------------------ Start to select subsets ------------------")
    if num_samples != 0:
        selected_data_idx = np.random.choice(len(compressed_train_y), num_samples,
                                             replace=False,
                                             p=scores / scores.sum())
        print("Start to select {} samples for a fair comparison.".format(len(selected_data_idx)))

    else:
        # for i in range(1, 10, 2):
        percent = i / 10.
        selected_data_idx = np.random.choice(len(compressed_train_y), int(percent * len(compressed_train_y)),
                                             replace=False,
                                             p=scores / scores.sum())

        print("Selected {} percent training data.".format(i * 10))
    his = train_with_original((train[0][selected_data_idx], train[1][selected_data_idx]), valid, test, net,
                              dataset, batch_size=batch_size, name="cl", stage=stage)
    his["size"] = len(selected_data_idx)
    history.append(his)

    return history


def run_wcl(train, valid, test, net, dataset, classes, batch_size=128, i=1, stage=1, num_samples=0):
    print("Now try to run the WCL algorithm")
    compressed_train_x, compressed_train_y = load_compressed_train_set(dataset, classes)
    # wcl = WCL()
    # wcl.fit_dataset(classes=classes, dataset=dataset)
    scores = np.load(os.path.join(os.getcwd(), "datasets", dataset, "cl_scores.npy"))
    selected_boundary_idx = np.load(os.path.join(os.getcwd(), "datasets", dataset, "cl_boundary_idx.npy"))
    # scores, selected_boundary_idx = wcl.fit(compressed_train_x, compressed_train_y, classes)
    print("Selected {} boundary instances.".format(len(selected_boundary_idx)))
    history = []
    # for i in range(1, 10, 2):
    num_full = num_samples
    num_samples = num_samples - len(selected_boundary_idx)
    if num_samples > 0:
        seleced_idx = selected_boundary_idx
        while num_samples != 0:
            diff_data_idx = np.random.choice(len(compressed_train_y), num_samples,
                                             replace=False, p=scores / scores.sum())
            seleced_idx = np.union1d(seleced_idx, diff_data_idx)
            num_samples = num_full - len(seleced_idx)
        print("Start to select {} samples for a fair comparison.".format(len(seleced_idx)))

    else:
        percent = i / 10.
        selected_data_idx = np.random.choice(len(compressed_train_y), int(percent * len(compressed_train_y)),
                                             replace=False, p=scores / scores.sum())
        print(
            "Select {:.2f} percent samples, {} overlapping with the pre-selected boundary samples".format(percent * 100,
                                                                                                          len(
                                                                                                              np.intersect1d(
                                                                                                                  selected_boundary_idx,
                                                                                                                  selected_data_idx))))
        seleced_idx = np.union1d(selected_boundary_idx, selected_data_idx)

    print("The unique selected subset size is: {}".format(len(seleced_idx)))
    his = train_with_original((train[0][seleced_idx], train[1][seleced_idx]), valid, test, net,
                              dataset, batch_size=batch_size, name="wcl", stage=stage)
    his["size"] = len(seleced_idx)
    history.append(his)

    return history


def run_wcl2(train, valid, test, net, dataset, classes, batch_size=128, i=1, stage=1, num_samples=0):
    # scale up boundary samples to score 1.
    print("Now try to run the WCL algorithm, by scaling up boundary sample scores")
    compressed_train_x, compressed_train_y = load_compressed_train_set(dataset, classes)
    # wcl = WCL()
    # wcl.fit_dataset(classes=classes, dataset=dataset)
    scores = np.load(os.path.join(os.getcwd(), "datasets", dataset, "cl_scores.npy"))
    selected_boundary_idx = np.load(os.path.join(os.getcwd(), "datasets", dataset, "cl_boundary_idx.npy"))
    # scores, selected_boundary_idx = wcl.fit(compressed_train_x, compressed_train_y, classes)
    print("Scale up the boundary sample scores to 1")
    scores[selected_boundary_idx] = 1
    # print("Selected {} boundary instances.".format(len(selected_boundary_idx)))
    history = []
    # for i in range(1, 10, 2):
    if num_samples == 0:
        num_samples = int(i / 10 * len(scores))
    print("Start to select {} samples for a fair comparison.".format(num_samples))
    seleced_idx = np.random.choice(len(compressed_train_y), num_samples,
                                   replace=False, p=scores / scores.sum())

    his = train_with_original((train[0][seleced_idx], train[1][seleced_idx]), valid, test, net,
                              dataset, batch_size=batch_size, name="wcl2", stage=stage)
    his["size"] = len(seleced_idx)
    history.append(his)

    return history


def run_wcl3(train, valid, test, net, dataset, classes, batch_size=128, i=1, stage=1, num_samples=0):
    # scale up boundary samples to score 1 from each subset to achieve balanced dataset.
    print("Now try to run the WCL algorithm, by scaling up boundary sample scores")
    compressed_train_x, compressed_train_y = load_compressed_train_set(dataset, classes)
    # wcl = WCL()
    # wcl.fit_dataset(classes=classes, dataset=dataset)
    scores = np.load(os.path.join(os.getcwd(), "datasets", dataset, "cl_scores.npy"))
    selected_boundary_idx = np.load(os.path.join(os.getcwd(), "datasets", dataset, "cl_boundary_idx.npy"))
    # scores, selected_boundary_idx = wcl.fit(compressed_train_x, compressed_train_y, classes)
    print("Scale up the boundary sample scores to 1")
    scores[selected_boundary_idx] = 1
    # print("Selected {} boundary instances.".format(len(selected_boundary_idx)))
    history = []
    # for i in range(1, 10, 2):
    if num_samples == 0:
        num_samples = int(i / 10 * len(scores))
    print("Start to select {} samples for a fair comparison at each subset.".format(num_samples))
    idx_list = []
    for i in range(classes):
        class_idx = np.argwhere(compressed_train_y == i)
        class_idx = class_idx.reshape(-1)
        if len(class_idx) >= int(num_samples / classes):
            seleced_idx = np.random.choice(len(class_idx), int(num_samples / classes),
                                           replace=False, p=scores[class_idx] / scores[class_idx].sum())
            idx_list.append(class_idx[seleced_idx])
        else:
            raise AttributeError("No enough samples for class {}".format(i + 1))
    seleced_idx = reduce(np.union1d, idx_list)
    print("Selected {} samples from all the classes".format(len(seleced_idx)))
    his = train_with_original((train[0][seleced_idx], train[1][seleced_idx]), valid, test, net,
                              dataset, batch_size=batch_size, name="wcl3", stage=stage)
    his["size"] = len(seleced_idx)
    history.append(his)

    return history
