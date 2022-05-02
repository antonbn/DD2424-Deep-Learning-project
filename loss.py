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
            # w shape: [Q]
            # Should work if w is now the shape [Q]
            return w[torch.argmax(Z_pred, dim=1)]
        return -torch.sum(v(Z_pred,w)*torch.sum(Z * np.log(Z_pred),dim=0))
    return loss