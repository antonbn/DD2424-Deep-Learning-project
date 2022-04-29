import numpy as np
import torch
from dataloaders import create_dataloader
from config import parse_configs
from matplotlib import pyplot as plt


if __name__ == '__main__':
    configs = parse_configs()
    train_loader = create_dataloader(configs.batch_size, configs.input_size, True, "train")
    val_loader = create_dataloader(configs.batch_size, configs.input_size, False, "val")
    test_loader = create_dataloader(configs.batch_size, configs.input_size, False, "test")
    print(torch.cuda.is_available())

    for bw, rgb in val_loader:
        _, axs = plt.subplots(8, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(bw, axs):
            ax.imshow(img.detach().numpy().squeeze(), cmap='gray')
        plt.show()
        _, axs = plt.subplots(8, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(rgb, axs):
            ax.imshow(np.transpose(img.detach().numpy(), [1, 2, 0]))
        plt.show()
        break
