import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import glob
from PIL import Image
import csv



class CustomDataSet(Dataset):
    def __init__(self, main_dir, input_size, mode):
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
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.transforms_no_normalize = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        file_name = os.path.basename(self.images[idx])
        file_name_no_ext = os.path.splitext(file_name)[0]
        image_no_normal = self.transforms_no_normalize(image)
        image = self.transforms(image)
        return image, image_no_normal, self.labels[file_name_no_ext]

def create_dataloader(batch_size, input_size, shuffle, mode):
    data = CustomDataSet("../ImageNet", input_size, mode)
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