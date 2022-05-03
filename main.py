import numpy as np
import torch
from dataloaders import create_dataloader, encode
from loss import CustomLoss
from config import parse_configs
from matplotlib import pyplot as plt
import pickle
from skimage import color, io
from annealed_mean import pred_to_ab, pred_to_ab_vec, pred_to_rgb_vec
from model import ConvNet
from tensorboardX import SummaryWriter
import tensorflow as tf
from tqdm import tqdm
from skimage import color
import torchvision

def load(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train(configs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA", torch.cuda.is_available())
    if device.type == "cpu":
        configs.batch_size = 2
    train_loader = create_dataloader(configs.batch_size, configs.input_size, True, "train_40000", "tree.p")
    val_loader = create_dataloader(configs.batch_size, configs.input_size, False, "val", "tree.p")
    #test_loader = create_dataloader(configs.batch_size, configs.input_size, False, "test", "tree.p")

    model = ConvNet().to(device)
    loss = CustomLoss("W_40000.npy", device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

    if configs.checkpoint:
        load(model, optimizer, configs.checkpoint)
    model.to(torch.double)


    train_log_dir = 'logs/tensorboard/' + configs.name
    train_summary_writer = SummaryWriter(train_log_dir)
    epochs = configs.num_epochs
    update_step = 0
    val_batch_X, val_batch_Z = encode(*next(iter(val_loader)), device)
    val_batch_im = pred_to_rgb_vec(val_batch_X, val_batch_Z, T=0.38)
    val_batch_im = torch.from_numpy(val_batch_im.transpose([0, 3, 1, 2]))
    val_batch_im = torchvision.utils.make_grid(val_batch_im)
    train_summary_writer.add_image('im_orig', val_batch_im, update_step)

    for e in tqdm(range(epochs)):
        running_loss = 0.0
        model.train()

        for X, Weights, ii in tqdm(train_loader, leave=False):
            X, Z = encode(X, Weights, ii, device)
            del ii
            del Weights
            Z_pred = model(X)
            J = loss(Z_pred, Z)
            optimizer.zero_grad()
            J.backward()
            optimizer.step()
            running_loss += J.item()
            if update_step % 100 == 0:
                running_loss /= 100
                train_summary_writer.add_scalar(f'info/Training loss', running_loss, update_step)
                """out = pred_to_ab_vec(y, 0.38)
                out = color.lab2rgb(np.concatenate((X_[:, :, :, 0][:, :, :, None], out), axis=-1))
                orig = color.lab2rgb(X_)
                train_summary_writer.add_image('imresult', image, update_step)"""
                running_loss = 0.0
            update_step += 1

    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, "/saved_models" + configs.name + ".tar")

if __name__ == '__main__':
    configs = parse_configs()
    train(configs)


