import numpy as np
import torch

def CustomLoss(w):
    """Returns the loss function, only here so that you can pass w to the real loss function"""
    def loss(Z_pred,Z):
        """Their custom loss function, unfinished DOES NOT WORK RIGHT NOW"""
        # Z shape: [x, Q, H, W]
        def v(Z_pred,w):
            """Their 'v' function as described in the paper, This is the unfinished part I believe, basically
            it should return the weight vector w_q where q = argmax Z_pred"""
            # Mainly unfinished because I got a brain aneurysm writing this code
            # Could easily be done with some loops, but the main problem is that it also needs to be quite fast I think?
            ZZ = torch.zeros(Z_pred.shape)
            arg_max = torch.argmax(torch.argmax(torch.argmax(Z_pred, dim=0), dim=2), dim=3)
            ZZ[torch.arange(Z_pred.shape[0]), arg_max, torch.arange(Z_pred.shape[2]), torch.arange(Z_pred.shape[3])] = 1
            ww = w[None,:,:,:]
            return torch.sum(w*ZZ, dim=0)
        return -torch.sum(v(Z_pred,w)*torch.sum(Z * np.log(Z_pred),dim=0))
    return loss