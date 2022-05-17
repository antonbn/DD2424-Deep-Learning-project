import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import glob
from skimage import color
from PIL import Image
import csv



class CustomDataSet(Dataset):
    def __init__(self, main_dir, input_size, mode, transform_extra):
        # Might want to create a new one that works on the output from our colorizer
        self.images = glob.glob(os.path.join(os.path.join(main_dir, mode), '*.JPEG'))
        self.mode = mode
        d = {}
        with open(main_dir + '/LOC_val_solution.csv', "r") as f:
            categories = csv.reader(f)
            next(categories, None)
            for lines in categories:
                d[lines[0]] = getnIDs(lines[1])
        self.labels = d
        self.input_size = input_size
        self.transform_extra = transform_extra
        if transform_extra == 'normalize':
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
        return image, self.labels[file_name_no_ext]

def create_dataloader(batch_size, input_size, shuffle, mode, transform_extra):
    data = CustomDataSet("../ImageNet", input_size, mode, transform_extra)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=1,
                                              pin_memory=True)
    return data_loader

def getnIDs(string):
    ids = []
    str_split = string.split(" ")
    n_ids = len(str_split)//5
    for i in range(n_ids):
        ids.append(str_split[i*5])
    return ids