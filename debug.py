import numpy as np
import torch
from dataloaders import create_dataloader, encode
from loss import CustomLoss
from config import parse_configs
import time
from tqdm import tqdm
from model import ConvNet
from matplotlib import pyplot as plt
import pickle
from skimage import color, io
from annealed_mean import pred_to_ab_vec, pred_to_rgb_vec


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = parse_configs()
    #train_loader = create_dataloader(configs.batch_size, configs.input_size, True, "train", "tree.p")
    val_loader = create_dataloader(4, configs.input_size, False, "val_4000", "tree.p", True)
    #test_loader = create_dataloader(configs.batch_size, configs.input_size, False, "test", "tree.p")
    if torch.cuda.is_available():
        print("CUDA")
    else:
        print("CPU")
    model = ConvNet(configs.custom_loss).to(device)
    model.to(torch.double)
    model.train()
    loss = CustomLoss("W_40000.npy", True, device)
    for k in range(2):
        start = time.time()
        for i, (X, Weights, ii) in tqdm(enumerate(val_loader)):
            X, Z = encode(X, Weights, ii, device)
            Z_pred = model(X)
            J = loss(Z_pred, Z)
            if i == 15:
                break
            """
            del ii
            del Weights
            plt.imshow(color.lab2rgb(X[0, :, :, :].numpy().copy().transpose([1, 2, 0])))
            plt.show()
            orig = pred_to_rgb_vec(X[:, 0, :, :].unsqueeze(1), y, T=0.38)
            plt.imshow(orig[0, :, :, :].numpy().transpose([1, 2, 0]))
            plt.show()
    
            pred = torch.rand((configs.batch_size, 322, configs.input_size, configs.input_size))
            J = loss(pred, y)
            _, axs = plt.subplots(8, 4, figsize=(12, 12))
            axs = axs.flatten()
            for img, ax in zip(X, axs):
                ax.imshow(img[0].detach().numpy(), cmap='gray')
            plt.show()
            _, axs = plt.subplots(8, 4, figsize=(12, 12))
            axs = axs.flatten()
            for img, ax in zip(X, axs):
                ax.imshow(img[1].detach().numpy(), cmap='gray')
            plt.show()
            break"""
        end = time.time()
        print("Time elapsed:", end - start)