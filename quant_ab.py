import numpy as np
from scipy.ndimage.interpolation import shift
from itertools import permutations
from skimage.color import rgb2lab
from matplotlib import pyplot as plt

def find_bins(displ = False):
    """Finds the bins in the quantized ab space that are useful and saves them to (some format when we've decided)
    by mapping (almost) all possible values of rgb colors to ab space."""
    # Just going to be run once so there's not that important to be efficient
    # 262 bins according to H
    # If you include neighbors, that increases it to 322 bins which is 9 more than they had
    # They don't describe how they did it, but I think this is a good attempt
    n = 128
    k = 3
    n_perm = int(np.math.factorial(n)/np.math.factorial(n-k))
    points = np.zeros([n_perm, k])
    for i, p in enumerate(permutations(range(n), k)):
        points[i, :] = np.array(p) / (n-1)

    points_ab = np.zeros([n_perm, 2])
    for i in range(n_perm):
        if i % n_perm and displ == 0:
            print(10 * i / n_perm)
        points_ab[i, :] = rgb2lab(points[i])[1:3]

    H, xedges, yedges = np.histogram2d(points_ab[:, 0], points_ab[:, 1], bins=np.arange(-115, 125, 10))

    # All relevant bins, i.e. bins that are used in the ab space
    for i in range(23):
        for j in range(23):
            if H[i, j] > 0:
                H[i, j] = 1

    # The neighbors of those bins
    G = H + shift(H, (1, 0), cval=0) + shift(H, (-1, 0), cval=0) + shift(H, (0, 1), cval=0) + shift(H, (0, -1), cval=0)

    # Make it binary
    for i in range(23):
        for j in range(23):
            if G[i, j] > 1e-2:
                G[i, j] = 1
            else:
                G[i, j] = 0

    # Make a dictionary so that you can map bin indices to the bin number (Q)
    # Might be better to store it some other way like the coordinates in ab space instead and do NN search
    count = 0
    dic = {}
    for i in range(23):
        for j in range(23):
            if G[i, j] == 1:
                dic.update({(i, j): count})
                count += 1

    # Uncomment when we know what we want to save and where
    # Might also want this function to return something
    # np.savez('AB_histogram', G=G, H=H, xedges=xedges, yedges=yedges)
    # np.save('AB_Q_dict', dic)

    if displ:
        X, Y = np.meshgrid(xedges, yedges)
        plt.gca().invert_yaxis()
        plt.pcolormesh(X, Y, H)
        plt.show()
        plt.gca().invert_yaxis()
        plt.pcolormesh(X, Y, G)
        plt.show()