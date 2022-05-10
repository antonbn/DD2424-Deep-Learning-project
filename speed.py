import numpy as np
from dataloaders import create_dataloader, encode
from tqdm import tqdm
import time
from scipy.stats import norm
import pickle
import torch
from loss import CustomLoss
from model import ConvNet
from annealed_mean import pred_to_rgb_vec

def measure():
    """Class rebalancing"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = create_dataloader(6, 224, False, "train_40000", "tree.p")
    with open("tree.p", 'rb') as pickle_file:
        tree = pickle.load(pickle_file)
    model = ConvNet().to(device)
    loss = CustomLoss("W_40000.npy", device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=.001)
    model.to(torch.double)

    start = time.time()

    end = 0
    for i, (X, Weights, ii) in enumerate(tqdm(train_dataloader)):
        # y [batch_size, 322, 224, 224]
        start2 = time.time()
        X, Z = encode(X, Weights, ii, device)
        Z_pred = model(X)
        J = loss(Z_pred, Z)
        optimizer.zero_grad()
        J.backward()
        optimizer.step()
        end += time.time() - start2
        if i == 50:
            break
    print("The whole loop")
    print((time.time() - start)/50)
    print("The gpu")
    print(end/50)






if __name__ == '__main__':
    measure()
