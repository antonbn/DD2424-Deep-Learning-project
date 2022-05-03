import numpy as np
import torch

class CustomLoss:
    def __init__(self, W_path, device):
        self.w = torch.from_numpy(np.load(W_path)).to(device)
    def __call__(self, Z_pred, Z):
        return -torch.sum(self.w[torch.argmax(Z_pred, axis=1)]*torch.sum(Z * torch.log(Z_pred), dim=1))/Z.shape[0]
