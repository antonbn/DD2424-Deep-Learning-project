import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import PIL.Image as Image
import glob

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform_bw, transform_rgb, mode):
        self.images = glob.glob(os.path.join(os.path.join(main_dir, mode), '**/*.JPEG'))
        self.mode = mode
        self.transform_bw = transform_bw
        self.transform_rgb = transform_rgb

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_loc = self.images[idx]
        image = Image.open(img_loc).convert("RGB")
        return self.transform_bw(image), self.transform_rgb(image)

def create_dataloader(batch_size, input_size, shuffle, mode):
    transform_rgb = transforms.Compose([transforms.CenterCrop(input_size),
                                        transforms.ToTensor()])
    
    transform_bw = transforms.Compose([transforms.CenterCrop(input_size),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5],
                                                         std=[0.5])])
    
    data = CustomDataSet("../ImageNet", transform_bw, transform_rgb, mode)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=4)
    return data_loader