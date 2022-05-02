import numpy as np
import torch
from dataloaders import create_dataloader, encode
from loss import CustomLoss
from config import parse_configs
from matplotlib import pyplot as plt


if __name__ == '__main__':
    configs = parse_configs()
    #train_loader = create_dataloader(configs.batch_size, configs.input_size, True, "train", "tree.p")
    val_loader = create_dataloader(configs.batch_size, configs.input_size, False, "val", "tree.p")
    #test_loader = create_dataloader(configs.batch_size, configs.input_size, False, "test", "tree.p")
    if torch.cuda.is_available():
        print("CUDA")
    else:
        print("CPU")
    loss = CustomLoss()

    for X, Weights, ii in val_loader:
        x, y = encode(X, Weights, ii)
        pred = torch.rand((configs.batch_size, 322, configs.input_size, configs.input_size))
        J = loss(pred, y)
        _, axs = plt.subplots(8, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(x, axs):
            ax.imshow(img[0].detach().numpy(), cmap='gray')
        plt.show()
        _, axs = plt.subplots(8, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(x, axs):
            ax.imshow(img[1].detach().numpy(), cmap='gray')
        plt.show()
        break
