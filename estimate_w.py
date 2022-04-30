import numpy as np
from torch.utils.data import Dataset, DataLoader


def CalculateW(train_dataloader):
    """Calculates the weights w which are used in the loss function,
    dangerously unfinished at this moment"""
    # For every image in training set:
    #   There's probably a better way to do it
    #   calculate which bin the color belongs to per pixel
    #   add that to a [Q,H,W] counter matrix, +1 if it belongs etc.
    #   (might be better to do it some other way tho)
    # Divide by the number of images to get the empirical dist.
    # Smooth the distribution with a gaussian kernel
    # Mix it with a uniform distribution
    # Normalize and save w as a [Q,H,W] matrix

    lamb = 1 / 2
    sigma = 5

    # image dimension
    p = np.zeros([322, 256, 256])

    for x, y in train_dataloader:
        # y contains the closest bin in ab space for each pixel (like one-hot encoded)
        # Must do some transform to get that :^)
        p += y.numpy()[0, :, :, :]

    p /= len(training_data)

    # TODO: Gaussian kernel here please, will probably have to go to 2d space instead of binned space?

    # calculate w
    w = np.reciprocal((1 - lamb) * p + lamb / p.shape[0])

    # normalize
    for i in range(256):
        for j in range(256):
            w[:, i, j] /= np.sum(p[:, i, j] * w[:, i, j])
    return w