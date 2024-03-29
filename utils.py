import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Reference: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

def load_data(train, test, batch_size, val_split=0.2):

    train = pd.read_csv(train, header=None).values
    test = pd.read_csv(test, header=None).values

    train = torch.utils.data.TensorDataset(torch.from_numpy(np.pad(train[:,:-1], ((0, 0), (0, 13)), 'constant')).float(),
                                           torch.from_numpy(train[:,-1]).long(),)

    test = torch.utils.data.TensorDataset(torch.from_numpy(np.pad(test[:,:-1], ((0, 0), (0, 13)), 'constant')).float(),
                                          torch.from_numpy(test[:,-1]).long())

    train_size = len(train)
    val_size = int(train_size * val_split)
    train_size = train_size - val_size

    train, val = torch.utils.data.random_split(train, [train_size, val_size])

    train_loader = DataLoader(train, batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def train_loop(dataloader, model, loss_fn, optimizer, device, scheduler= None, verbose = True):
    
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    y_true = []
    y_pred = []

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.argmax(1).cpu().numpy())

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            if verbose:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"Learning rate: {scheduler.get_last_lr()[0]}")

    train_loss /= len(dataloader)
    train_f1_score = f1_score(y_true, y_pred, average='macro')
    return train_f1_score, train_loss


def evaluate(dataloader, model, loss_fn, device):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.argmax(1).cpu().numpy())

    test_loss /= num_batches
    test_f1_score = f1_score(y_true, y_pred, average='macro')
    return test_f1_score, test_loss
