import numpy as np


def to_gray(ab):
    return np.zeros_like(ab)


def to_random(ab, auc_dataloader):
    for X_train, _, _, _ in auc_dataloader:
        rnd_img_indices = np.random.choice(np.arange(X_train.shape[0]), X_train.shape[0])
        return ab[rnd_img_indices]
