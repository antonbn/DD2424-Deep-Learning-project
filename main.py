import numpy as np
import torch
from dataloaders import create_dataloader, encode
from loss import CustomLoss
from config import parse_configs
from matplotlib import pyplot as plt
import pickle
from skimage import color, io
from annealed_mean import pred_to_ab, pred_to_ab_vec


if __name__ == '__main__':
    configs = parse_configs()
    #train_loader = create_dataloader(configs.batch_size, configs.input_size, True, "train", "tree.p")
    val_loader = create_dataloader(4, configs.input_size, False, "val", "tree.p")
    #test_loader = create_dataloader(configs.batch_size, configs.input_size, False, "test", "tree.p")
    if torch.cuda.is_available():
        print("CUDA")
    else:
        print("CPU")
    loss = CustomLoss()

    for X, Weights, ii in val_loader:
        X, y = encode(X, Weights, ii)
        del ii
        del Weights
        X_ = X.numpy().transpose([0, 2, 3, 1])
        out = pred_to_ab_vec(y, 0.38)
        out = color.lab2rgb(np.concatenate((X_[:, :, :, 0][:, :, :, None], out), axis=-1))
        orig = color.lab2rgb(X_)
        plt.imshow(orig[0, :, :, :])
        plt.show()
        plt.imshow(out[0, :, :, :])
        plt.show()

        pred = torch.rand((4, 322, configs.input_size, configs.input_size))
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
