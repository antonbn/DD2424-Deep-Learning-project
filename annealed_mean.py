import numpy as np
import torch
import pickle
from skimage import color

def pred_to_ab(Z_pred, T=.38):
    with open("tree.p", 'rb') as pickle_file:
        tree = pickle.load(pickle_file)

    batch_size, Q, H, W = Z_pred.shape

    img_ab = np.zeros((batch_size, H, W, 2))

    for i in range(batch_size):

        for h in range(H):
            for w in range(W):
                p_interpolation = torch.exp(torch.log(Z_pred[i, :, h, w]) / T)
                p_interpolation /= torch.sum(p_interpolation)

                a = np.sum(np.multiply(p_interpolation.numpy(), tree.data[:, 0]))
                b = np.sum(np.multiply(p_interpolation.numpy(), tree.data[:, 1]))
                img_ab[i, h, w, 0] = a
                img_ab[i, h, w, 1] = b


    return img_ab

def pred_to_ab_vec(Z, T):
    with open("tree.p", 'rb') as pickle_file:
        tree = pickle.load(pickle_file)
    annealed = torch.exp(torch.log(Z)/T)/torch.sum(torch.exp(torch.log(Z)/T), axis=1)[:, None]
    annealed = np.stack((np.sum(tree.data[None, :, None, 0, None] * annealed.numpy(), axis=1),
              np.sum(tree.data[None, :, None, 1, None] * annealed.numpy(), axis=1)))
    return annealed.transpose([1, 2, 3, 0])

def pred_to_rgb_vec(X, Z, T=0.38):
    X = X.cpu().numpy()
    Z = pred_to_ab_vec(Z.cpu(), T)
    lab = np.concatenate((X.transpose([0, 2, 3, 1]), Z), axis=-1)
    rgb = color.lab2rgb(lab)
    return rgb

