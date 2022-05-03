import numpy as np
from dataloaders import create_dataloader, encode
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import time

def CalculateSaveW():
    """Class rebalancing"""
    train_dataloader = create_dataloader(1, 224, False, "train_40000", "tree.p")
    lamb = 0.5
    sigma = 5

    import time
    start = time.process_time()
    # ab color distribution (now int, but it will get transformed into the correct shape 1D)
    p = 0
    for i, (X, Weights, ii) in enumerate(tqdm(train_dataloader)):
        # y [batch_size, 322, 224, 224]
        X, y = encode(X, Weights, ii)
        p += y.mean(axis=(0, 2, 3))
        del ii
        del Weights
    print(time.process_time() - start)

    p /= len(train_dataloader)

    # smooth with gaussian filter
    p = p.cpu().numpy()
    p_smooth = gaussian_filter1d(p, sigma)
    # mix with uniform
    w = 1/((1 - lamb) * p_smooth + lamb / p.shape[0])

    # normalize
    w = w / np.dot(w, p)
    np.save("W.npy", w)


if __name__ == '__main__':
    CalculateSaveW()