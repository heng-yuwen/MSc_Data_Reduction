'''Train CIFAR10 with PyTorch.'''
import os

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset

from lib.data_loader import load_cifar10
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
    return acc


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
    if acc > best_acc:
        print('Saving with better validation accuracy model: {:.3f}..'.format(acc))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    else:
        state = None

    return acc, best_acc, state


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
    return acc


def check_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def train_with_original(train, valid, test, net, dataset, batch_size=128):
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
        print("Use GPU to train the network.")

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

    history = {"acc": [], "val_acc": [], "test_acc": 0}

    for epoch in range(350):
        if epoch < 150:
            optimizer = optim.SGD(net.parameters(), lr=0.1,
                                  momentum=0.9, weight_decay=5e-4)
        elif epoch < 250:
            optimizer = optim.SGD(net.parameters(), lr=0.01,
                                  momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = optim.SGD(net.parameters(), lr=0.001,
                                  momentum=0.9, weight_decay=5e-4)

        train_acc = train_epoch(trainloader, epoch, net, device, optimizer, criterion)
        valid_acc, best_acc, state = valid_epoch(validloader, epoch, net, device, criterion, best_acc)
        if state:
            best_state = state
        history["acc"].append(train_acc)
        history["val_acc"].append(valid_acc)

    test_acc = test_epoch(testloader, net, device, criterion)
    history["test_acc"] = test_acc

    print("Saving best model parameters with validation acc: {}".format(best_acc))
    torch.save(best_state, os.path.join(os.getcwd(), "models", dataset, "whole_training_set_ckpt.pth"))

    return history


def run_pop(dataset, classes):
    compressed_train_x, compressed_train_y = load_compressed_train_set(dataset, classes)
    pop = POP()

    raise NotImplementedError("Implement first")
