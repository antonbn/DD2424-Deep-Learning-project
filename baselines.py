import numpy as np


def to_gray(X):
    return np.zeros_like(X)[:, :, :, 1:]


def to_random(X, train_dataloader):
    for X_train, _, _ in train_dataloader:
        rnd_img_indices = np.random.choice(np.arange(X_train.shape[0]), X_train.shape[0])
        return X[rnd_img_indices, :, :, 1:]
