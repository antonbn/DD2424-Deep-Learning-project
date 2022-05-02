import numpy as np
import torch
from dataloaders import create_dataloader
from loss import CustomLoss
from config import parse_configs
from matplotlib import pyplot as plt


if __name__ == '__main__':
    configs = parse_configs()
    #train_loader = create_dataloader(configs.batch_size, configs.input_size, True, "train", tree_path)
    val_loader = create_dataloader(configs.batch_size, configs.input_size, False, "val", "tree.p")
    #test_loader = create_dataloader(configs.batch_size, configs.input_size, False, "test", tree_path)
    print(torch.cuda.is_available())
    loss = CustomLoss()

    for bw, y in val_loader:
        pred = torch.rand((configs.batch_size, 6, configs.input_size, configs.input_size))
        J = loss(pred, y)
        _, axs = plt.subplots(8, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(bw, axs):
            ax.imshow(img[0].detach().numpy().squeeze(), cmap='gray')
        plt.show()
        _, axs = plt.subplots(8, 4, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(bw, axs):
            ax.imshow(img[1].detach().numpy(), cmap='gray')
        plt.show()
        break
