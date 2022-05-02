import numpy as np
import torch
import pickle


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
