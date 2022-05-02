import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import glob
from skimage import color, io
from PIL import Image
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm
import pickle

class CustomDataSet(Dataset):
    def __init__(self, main_dir, input_size, mode, tree_path):
        self.images = glob.glob(os.path.join(os.path.join(main_dir, mode), '*.JPEG'))
        self.mode = mode
        self.input_size = input_size
        self.n_neighbours = 5
        with open(tree_path, 'rb') as pickle_file:
            self.tree = pickle.load(pickle_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        x = transforms.Resize((self.input_size, self.input_size))(image)
        rgb = np.array(x)
        lab = color.rgb2lab(rgb)
        lab = lab.transpose([2, 0, 1])
        dist, ii = self.tree.query(lab[1:3].reshape(2, self.input_size**2).T, self.n_neighbours)
        x = torch.from_numpy(lab)
        weights = norm.pdf(dist, loc=0, scale=5)
        if self.n_neighbours > 1:
            y = torch.sum(torch.eye(self.tree.n)[ii] * weights[:, :, None], axis=1)/weights.sum(axis=1)[:, None]
        else:
            y = torch.eye(self.tree.n)[ii]
        return x, y.T.reshape(self.tree.n, 224, 224)

def create_dataloader(batch_size, input_size, shuffle, mode, tree_path):
    data = CustomDataSet("../ImageNet", input_size, mode, tree_path)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=4,
                                              pin_memory=True)
    return data_loader