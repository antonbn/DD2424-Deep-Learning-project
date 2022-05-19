import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import glob
from skimage import color
from PIL import Image
import pickle
from scipy.stats import norm

class CustomDataSet(Dataset):
    def __init__(self, main_dir, input_size, mode, tree_path):
        self.images = glob.glob(os.path.join(os.path.join(main_dir, mode), '*.JPEG'))
        self.mode = mode
        self.input_size = input_size
        self.n_neighbours = 5
        self.q = 322
        with open(tree_path, 'rb') as pickle_file:
            self.tree = pickle.load(pickle_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        rgb = transforms.Resize((self.input_size, self.input_size))(image)
        rgb = transforms.RandomHorizontalFlip()(rgb)
        lab = color.rgb2lab(rgb)
        lab = lab.transpose([2, 0, 1])
        dist, ii = self.tree.query(lab[1:3].reshape(2, self.input_size**2).T, self.n_neighbours)
        x = torch.from_numpy(lab)
        weights = norm.pdf(dist, loc=0, scale=5)
        if self.n_neighbours > 1:
            weights = weights / weights.sum(axis=-1)[:, None]
        file_name = os.path.basename(self.images[idx])
        file_name_no_ext = os.path.splitext(file_name)[0]
        return torch.unsqueeze(x[0], 0), weights, ii, file_name_no_ext.split('_')[0]

class CustomDataSetVGG(Dataset):
    def __init__(self, main_dir, input_size, mode, transform_extra):
        # Might want to create a new one that works on the output from our colorizer
        self.images = glob.glob(os.path.join(os.path.join(main_dir, mode), '*.JPEG'))
        self.mode = mode
        self.input_size = input_size
        self.transform_extra = transform_extra
        if transform_extra == 'normalize' or transform_extra == 'orig':
            self.transforms = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif transform_extra == 'random' or transform_extra == 'grayscale':
            self.transforms = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size))])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform_extra == 'random':
            image = Image.open(self.images[idx]).convert('RGB')
            idxx = np.random.randint(self.__len__())
            image2 = Image.open(self.images[idxx]).convert('RGB')
            image = self.transforms(image)
            image2 = self.transforms(image2)
            lab = color.rgb2lab(image)
            lab2 = color.rgb2lab(image2)
            lab[:,:,1:3] = lab2[:,:,1:3]
            hmm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            rgb = color.lab2rgb(lab)
            image = hmm(rgb).to(torch.float32)
        elif self.transform_extra == 'grayscale':
            image = Image.open(self.images[idx]).convert('RGB')
            image = self.transforms(image)
            lab = color.rgb2lab(image)
            lab[:,:,1:3] = np.zeros_like(lab[:,:,1:3])
            hmm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            rgb = color.lab2rgb(lab)
            image = hmm(rgb).to(torch.float32)
        else:
            image = Image.open(self.images[idx]).convert('RGB')
            image = self.transforms(image)
        file_name = os.path.basename(self.images[idx])
        file_name_no_ext = os.path.splitext(file_name)[0]
        return image, file_name_no_ext.split('_')[0]

def create_dataloaderVGG(batch_size, input_size, shuffle, mode, transform_extra):
    data = CustomDataSetVGG("../ImageNet", input_size, mode, transform_extra)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=1,
                                              pin_memory=True)
    return data_loader

def create_dataloader_label(batch_size, input_size, shuffle, mode, tree_path):
    data = CustomDataSet("../ImageNet", input_size, mode, tree_path)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=2,
                                              pin_memory=True)
    return data_loader

def getnIDs(string):
    ids = []
    str_split = string.split(" ")
    n_ids = len(str_split)//5
    for i in range(n_ids):
        ids.append(str_split[i*5])
    return ids