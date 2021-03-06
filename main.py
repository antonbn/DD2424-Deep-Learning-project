import torch
from dataloaders import create_dataloader, encode
from loss import CustomLoss
from config import parse_configs
from annealed_mean import pred_to_rgb_vec, lab_to_rgb
from model import ConvNet
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision
import os

def load(model, optimizer, name):
    checkpoint = torch.load(os.path.join("saved_models", name))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train(configs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA", torch.cuda.is_available())
    if device.type == "cpu":
        configs.batch_size = 1
    train_loader = create_dataloader(configs.batch_size, configs.input_size, True, "train_40000", "tree.p", configs.custom_loss)
    val_loader = create_dataloader(configs.batch_size, configs.input_size, False, "val_4000", "tree.p", configs.custom_loss)

    model = ConvNet(configs.custom_loss).to(device)
    model.to(torch.double)
    if configs.custom_loss:
        loss = CustomLoss("W_40000.npy", configs.use_weights, device)
    else:
        loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=.001)

    update_step = 0
    e_load = 0
    if configs.checkpoint:
        load(model, optimizer, configs.checkpoint)


    train_log_dir = 'logs/tensorboard/' + configs.name
    train_summary_writer = SummaryWriter(train_log_dir)
    epochs = configs.num_epochs
    if configs.custom_loss:
        val_batch_X, val_batch_Z = encode(*next(iter(val_loader)), device)
        val_batch_rgb = pred_to_rgb_vec(val_batch_X, torch.log(val_batch_Z), device, T=0.38)
    else:
        val_batch_lab = next(iter(val_loader))[0].to(device).to(torch.double)
        val_batch_rgb = lab_to_rgb(val_batch_lab)
    val_batch_im = torchvision.utils.make_grid(val_batch_rgb)
    train_summary_writer.add_image('im_orig', val_batch_im, update_step)

    running_loss = 0.0
    for e in tqdm(range(epochs)):
        for X, Weights, ii in tqdm(train_loader, leave=False):
            model.train()
            if configs.custom_loss:
                X, Z = encode(X, Weights, ii, device)
            else:
                X = X.to(device).to(torch.double)
            del ii
            del Weights
            if configs.custom_loss:
                Z_pred = model(X)
                J = loss(Z_pred, Z)
            else:
                Y_pred = model(X[:, 0, :, :].unsqueeze(1))
                J = loss(Y_pred, X[:, 1:, :, :])
            optimizer.zero_grad()
            J.backward()
            optimizer.step()
            running_loss += J.item()
            if update_step % configs.train_loss_fr == 0:
                running_loss /= configs.train_loss_fr
                model.eval()
                with torch.no_grad():
                    if configs.custom_loss:
                        val_batch_Z = model(val_batch_X)
                        val_batch_im = pred_to_rgb_vec(val_batch_X, val_batch_Z, device, T=0.38)
                    else:
                        val_batch_im = model(val_batch_lab[:, 0, :, :].unsqueeze(1))
                        val_batch_im = lab_to_rgb(torch.cat((val_batch_lab[:, 0, :, :].unsqueeze(1), val_batch_im), axis=1))
                    val_batch_im = torchvision.utils.make_grid(val_batch_im)
                train_summary_writer.add_scalar(f'info/Training loss', running_loss, update_step)
                train_summary_writer.add_image('imresult', val_batch_im, update_step)
                running_loss = 0.0
            if update_step % configs.val_loss_fr == 0:
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for X, Weights, ii in tqdm(val_loader, leave=False):
                        if configs.custom_loss:
                            X, Z = encode(X, Weights, ii, device)
                        else:
                            X = X.to(device).to(torch.double)
                        del ii
                        del Weights
                        if configs.custom_loss:
                            Z_pred = model(X)
                            J = loss(Z_pred, Z)
                        else:
                            Y_pred = model(X[:, 0, :, :].unsqueeze(1))
                            J = loss(Y_pred, X[:, 1:, :, :])
                        val_loss += J.item()
                train_summary_writer.add_scalar(f'info/Validation loss', val_loss, update_step)
            update_step += 1

        state = {'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, "saved_models/" + configs.name + "_" + str(e+e_load) +".tar")

if __name__ == '__main__':
    configs = parse_configs()
    train(configs)


