import numpy as np
import os
import torch
import pickle
from main import load
from model import ConvNet
from tqdm import tqdm
from config import parse_configs
from annealed_mean import pred_to_ab_vec, pred_to_rgb_vec
from dataloaders import create_dataloader
from dataloaders import encode
from torchvision.utils import save_image, make_grid
import torchvision.transforms as T
from PIL import Image


configs = parse_configs()
transform = T.ToPILImage()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_loader = create_dataloader(configs.batch_size, configs.input_size, False, "val_4000", "tree.p", configs.custom_loss)
model = ConvNet(True).to(device)
model.to(torch.double)
optimizer_nn1 = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=.001)
load(model, optimizer_nn1, 'cars_1_NN_21.tar')
model.eval()

for i, (X, Weights, ii) in enumerate(tqdm(val_loader, leave=False)):
    X, Z = encode(X, Weights, ii, device)
    Z_pred = model(X)
    val_batch_im = pred_to_rgb_vec(X, Z_pred, device, T=0.38)
    val_batch_im = make_grid(val_batch_im)
    val_batch_GT = pred_to_rgb_vec(X, torch.log(Z), device, T=0.38)
    val_batch_GT = make_grid(val_batch_GT)
    im = transform(val_batch_GT)
    im.save("images/%s_%s.png" % ("GT", str(i)))
    im = transform(val_batch_im)
    im.save("images/%s_%s.png" % ("fake", str(i)))
