import numpy as np
from dataloaders import create_dataloader, encode
from tqdm import tqdm
import time
from scipy.stats import norm
import pickle
import torch

def CalculateSaveW():
    """Class rebalancing"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = create_dataloader(6, 224, False, "sports_cars/train", "tree.p")
    lamb = 0.5
    sigma = 5
    with open("tree.p", 'rb') as pickle_file:
        tree = pickle.load(pickle_file)

    start = time.process_time()
    start2 = time.time()
    # ab color distribution (now int, but it will get transformed into the correct shape 1D)
    p = 0
    for i, (X, Weights, ii) in enumerate(tqdm(train_dataloader)):
        # y [batch_size, 322, 224, 224]
        X, y = encode(X, Weights, ii, device)
        p += y.mean(axis=(0, 2, 3))
        #if i == 100:
            #break
    print(time.process_time() - start)
    print(time.time() - start2)

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
    np.save("p_sports_cars.npy", p)
    np.save("p_smooth_sports_cars.npy", p_smooth)
    np.save("W_sports_cars.npy", w)


if __name__ == '__main__':
    CalculateSaveW()
