import numpy as np
import torch
from dataloaders import create_dataloader, encode
from loss import CustomLoss
from config import parse_configs
from matplotlib import pyplot as plt
import pickle
from skimage import color, io
from annealed_mean import pred_to_ab_vec, pred_to_rgb_vec


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = parse_configs()
    #train_loader = create_dataloader(configs.batch_size, configs.input_size, True, "train", "tree.p")
    val_loader = create_dataloader(4, configs.input_size, False, "val_4000", "tree.p")
    #test_loader = create_dataloader(configs.batch_size, configs.input_size, False, "test", "tree.p")
    if torch.cuda.is_available():
        print("CUDA")
    else:
        print("CPU")
    loss = CustomLoss("W_40000.npy", device)

    for X, Weights, ii in val_loader:
        X, y = encode(X, Weights, ii, device)
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
        break
