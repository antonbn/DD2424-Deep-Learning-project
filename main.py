import numpy as np
import torch
from dataloaders import create_dataloader, encode
from loss import CustomLoss
from config import parse_configs
from matplotlib import pyplot as plt
import pickle
from skimage import color, io
from annealed_mean import pred_to_ab_vec, pred_to_rgb_vec
from model import ConvNet
from tensorboardX import SummaryWriter
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
        configs.batch_size = 1
    train_loader = create_dataloader(configs.batch_size, configs.input_size, True, "train_40000", "tree.p")
    val_loader = create_dataloader(configs.batch_size, configs.input_size, False, "val_4000", "tree.p")
    #test_loader = create_dataloader(configs.batch_size, configs.input_size, False, "test", "tree.p")

    model = ConvNet().to(device)
    loss = CustomLoss("W_40000.npy", device)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=.001)

    if configs.checkpoint:
        load(model, optimizer, configs.checkpoint)
    model.to(torch.double)


    train_log_dir = 'logs/tensorboard/' + configs.name
    train_summary_writer = SummaryWriter(train_log_dir)
    epochs = configs.num_epochs
    update_step = 0
    val_batch_X, val_batch_Z = encode(*next(iter(val_loader)), device)
    val_batch_im = pred_to_rgb_vec(val_batch_X, val_batch_Z, device, T=0.38)
    val_batch_im = torchvision.utils.make_grid(val_batch_im)
    train_summary_writer.add_image('im_orig', val_batch_im, update_step)

    for e in tqdm(range(epochs)):
        running_loss = 0.0
        for X, Weights, ii in tqdm(train_loader, leave=False):
            model.train()
            X, Z = encode(X, Weights, ii, device)
            del ii
            del Weights
            Z_pred = model(X)
            J = loss(Z_pred, Z)
            optimizer.zero_grad()
            J.backward()
            optimizer.step()
            running_loss += J.item()
            if update_step % configs.train_loss_fr == 0:
                running_loss /= configs.train_loss_fr
                model.eval()
                with torch.no_grad():
                    val_batch_Z = model(val_batch_X)
                    val_batch_im = pred_to_rgb_vec(val_batch_X, val_batch_Z, device, T=0.38)
                    val_batch_im = torchvision.utils.make_grid(val_batch_im)
                train_summary_writer.add_scalar(f'info/Training loss', running_loss, update_step)
                train_summary_writer.add_image('imresult', val_batch_im, update_step)
                running_loss = 0.0
            if update_step % configs.val_loss_fr == 0:
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for X, Weights, ii in tqdm(val_loader, leave=False):
                        X, Z = encode(X, Weights, ii, device)
                        del ii
                        del Weights
                        Z_pred = model(X)
                        J = loss(Z_pred, Z)
                        val_loss += J.item()
                train_summary_writer.add_scalar(f'info/Validation loss', val_loss, update_step)
            update_step += 1

        state = {'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, "/saved_models" + configs.name + "_" + str(e) +"_.tar")

if __name__ == '__main__':
    configs = parse_configs()
    train(configs)


