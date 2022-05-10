import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import glob
from skimage import color
from PIL import Image
from scipy.stats import norm
import pickle
import json


class CustomDataSet(Dataset):
    def __init__(self, main_dir, input_size, mode):
        self.images = glob.glob(os.path.join(os.path.join(main_dir, mode), '*/*.JPEG'))
        self.mode = mode
        with open(main_dir + '/Labels.json', "r") as f:
            categories = [s.strip() for s in f.readlines()]
        A = categories[0].split('"')
        d = {}
        for i in range(100):
            d[A[1 + i * 4]] = A[3 + i * 4].split(",")[0]
        self.labels = d
        self.input_size = input_size
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label_label = self.images[idx].split('\\')[-2]
        image = self.transforms(image)
        return image, self.labels[label_label]

def create_dataloader(batch_size, input_size, shuffle, mode):
    data = CustomDataSet("../ImageNet", input_size, mode)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=1,
                                              pin_memory=True)
    return data_loader


def encode(X, Weights, ii, device):
    """
    :return:
    X [batch_size, 1, H, W]
    Y [batch_size, 322, H, W]
    """
    n = 322
    X, Weights, ii = X.to(device), Weights.to(device), ii.to(device)
    if len(Weights.shape) > 2:
        Y = torch.zeros((X.shape[0], X.shape[-1]*X.shape[-2], n)).to(device)
        I = torch.arange(Weights.shape[0])[:, None, None].to(device)
        I2 = torch.arange(Weights.shape[1])[None, :, None].to(device)
        Y[I, I2, ii] = Weights.float()
        #Y = torch.sum(torch.eye(n)[ii].to(device) * Weights[:, :, :, None], axis=-2)
    else:
        Y = torch.eye(n)[ii].to(device)
    return X.double(), Y.transpose(2, 1).reshape(X.shape[0], n, 224, 224).double()