import torch
import pickle
from util import lab2rgb, rgb2lab
from skimage import color
import numpy as np

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

def pred_to_ab_vec(Z, T, device):
    with open("tree.p", 'rb') as pickle_file:
        tree = pickle.load(pickle_file)
    data0 = torch.from_numpy(tree.data)[None, :, None, 0, None].to(device)
    data1 = torch.from_numpy(tree.data)[None, :, None, 1, None].to(device)
    annealed = torch.exp(torch.log(Z)/T)/torch.sum(torch.exp(torch.log(Z)/T), axis=1)[:, None]
    annealed = torch.stack((torch.sum(data0 * annealed, axis=1),
              torch.sum(data1 * annealed, axis=1)))
    return torch.transpose(annealed, 0, 1)

def pred_to_rgb_vec(X, Z, device, T=0.38):
    X = X.detach()
    Z = pred_to_ab_vec(Z.detach(), T, device)
    lab = torch.cat((X, Z), axis=1).detach().cpu().numpy()
    rgb = color.lab2rgb(lab.transpose([0, 2, 3, 1]))
    rgb = torch.from_numpy(rgb.transpose([0, 3,  1, 2]))
    return rgb

