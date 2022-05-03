import numpy as np
from dataloaders import create_dataloader, encode
from tqdm import tqdm
import time
from scipy.stats import norm
import pickle

def CalculateSaveW():
    """Class rebalancing"""
    train_dataloader = create_dataloader(1, 224, False, "train_40000", "tree.p")
    lamb = 0.5
    sigma = 5
    with open("tree.p", 'rb') as pickle_file:
        tree = pickle.load(pickle_file)

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

    p_smooth = np.zeros_like(p)
    for i in range(322):
        weights = norm.pdf(tree.data, loc=tree.data[i], scale=sigma)
        weights = weights[:, 0]*weights[:, 1]
        weights = weights/weights.sum()
        p_smooth[i] = np.dot(p, weights)
    # mix with uniform
    w = 1/((1 - lamb) * p_smooth + lamb / p_smooth.shape[0])

    # normalize
    w = w / np.dot(w, p_smooth)
    np.save("p_40000.npy", p)
    np.save("p_smooth_40000.npy", p_smooth)
    np.save("W_40000.npy", w)


if __name__ == '__main__':
    CalculateSaveW()
