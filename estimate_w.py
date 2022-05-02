import numpy as np
from dataloaders import create_dataloader
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

def CalculateSaveW():
    """Class rebalancing"""
    train_dataloader = create_dataloader(64, 224, False, "train", "tree.p")
    lamb = 0.5
    sigma = 5

    # ab color distribution (now int, but it will get transformed into the correct shape 1D)
    p = 0
    for x, y in tqdm(train_dataloader):
        # y [batch_size, 322, 224, 224]
        p += y.numpy().mean(axis=(0, 2, 3))

    p /= len(train_dataloader)

    # smooth with gaussian filter
    p_smooth = gaussian_filter1d(p, sigma)
    # mix with uniform
    w = 1/((1 - lamb) * p_smooth + lamb / p.shape[0])

    # normalize
    w = w / np.dot(w, p)
    np.save("W.npy", w)


if __name__ == '__main__':
    CalculateSaveW()