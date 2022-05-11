import torch
import pickle
from skimage import color
import numpy as np



def pred_to_ab_vec(Z, T, device):
    with open("tree.p", 'rb') as pickle_file:
        tree = pickle.load(pickle_file)
    data0 = torch.from_numpy(tree.data)[None, :, None, 0, None].to(device)
    data1 = torch.from_numpy(tree.data)[None, :, None, 1, None].to(device)
    annealed = torch.exp(Z/T)/torch.sum(torch.exp(Z/T), axis=1)[:, None]
    annealed = torch.stack((torch.sum(data0 * annealed, axis=1),
              torch.sum(data1 * annealed, axis=1)))
    return torch.transpose(annealed, 0, 1)

def pred_to_rgb_vec(X, Z, device, T=0.38):
    X = X.detach()
    Z = pred_to_ab_vec(Z.detach(), T, device)
    lab = torch.cat((X, Z), axis=1).detach().cpu().numpy()
    rgb = color.lab2rgb(lab.transpose([0, 2, 3, 1]))
    rgb = (np.clip(rgb, 0, 1)*255).astype('uint8')
    rgb = torch.from_numpy(rgb.transpose([0, 3,  1, 2]))
    return rgb

