import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import glob
from skimage import color
from PIL import Image
from scipy.stats import norm
import pickle


class CustomDataSet(Dataset):
    def __init__(self, main_dir, input_size, mode, tree_path, custom_loss):
        self.images = glob.glob(os.path.join(os.path.join(main_dir, mode), '*.JPEG'))
        self.mode = mode
        self.input_size = input_size
        self.n_neighbours = 5
        self.q = 322
        self.custom_loss = custom_loss
        with open(tree_path, 'rb') as pickle_file:
            self.tree = pickle.load(pickle_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        rgb = transforms.Resize((self.input_size, self.input_size))(image)
        if "val" not in self.mode:
            rgb = transforms.RandomHorizontalFlip()(rgb)
        lab = color.rgb2lab(rgb)
        if self.custom_loss:
            lab = lab.transpose([2, 0, 1])
            dist, ii = self.tree.query(lab[1:3].reshape(2, self.input_size**2).T, self.n_neighbours)
            x = torch.from_numpy(lab)
            weights = norm.pdf(dist, loc=0, scale=5)
            if self.n_neighbours > 1:
                weights = weights / weights.sum(axis=-1)[:, None]
            return torch.unsqueeze(x[0], 0), weights, ii
        else:
            return transforms.functional.to_tensor(lab), 0, 0

    """def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        x = transforms.Resize((self.input_size, self.input_size))(image)
        rgb = np.array(x)
        lab = color.rgb2lab(rgb)
        lab = lab.transpose([2, 0, 1])
        dist, ii = self.tree.query(lab[1:3].reshape(2, self.input_size ** 2).T, self.n_neighbours)
        x = torch.from_numpy(lab)
        if self.n_neighbours == 1:
            Y = torch.eye(self.q)[ii]
        else:
            weights = norm.pdf(dist, loc=0, scale=5)
            weights = weights / weights.sum(axis=-1)[:, None]
            Y = torch.zeros((224 * 224, self.q))
            I = torch.arange(weights.shape[0])[:, None]
            Y[I, ii] = torch.from_numpy(weights).float()

        return torch.unsqueeze(x[0], 0).double(), Y.T.reshape(self.q, 224, 224).double()"""


def create_dataloader(batch_size, input_size, shuffle, mode, tree_path, custom_loss):
    data = CustomDataSet("../ImageNet", input_size, mode, tree_path, custom_loss)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=2,
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
