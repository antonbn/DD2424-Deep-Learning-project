import numpy as np
import torch
from VGG_dataloader import create_dataloaderVGG, create_dataloader_label
from dataloaders import encode
from tqdm import tqdm
from config import parse_configs
from annealed_mean import pred_to_rgb_vec
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from main import load
import os
from model import ConvNet

def VGG_download(path):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    torch.save(model, path)
    return model

def VGG_load(path):
    return torch.load(path)

def VGG_eval(configs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'pretrained_VGG.pt'
    if not os.path.exists(path):
        VGGnet = VGG_download(path)
    else:
        VGGnet = VGG_load(path)

    VGGnet.eval()

    with open('..\ImageNet\LOC_synset_mapping.txt', "r") as f:
        categories = [s.strip() for s in f.readlines()]
    categories = [x.split(" ")[0] for x in categories]

    models_to_test = ['orig', 'random', 'grayscale','cars_full_35.tar', 'cars_no_weights_44.tar', 'cars_L2_64.tar', 'cars_1_NN_21.tar']

    accuracies = np.zeros(len(models_to_test))

    for acc_index, model_name in enumerate(models_to_test):

        if model_name == 'orig' or model_name == 'random' or model_name == 'grayscale':
            val_loader = create_dataloaderVGG(1, configs.input_size, False, "sports_cars/val", model_name)
            correct = 0
            n_guesses = 0
            for x, y in tqdm(val_loader, total=len(val_loader)):
                with torch.no_grad():
                    output = VGGnet(x)
                _, guess_catid = torch.topk(torch.nn.functional.softmax(output[0], dim=0), 1)
                n_guesses += 1
                if categories[guess_catid] in y[0]:
                    correct += 1
            accuracies[acc_index] = correct/n_guesses

        else:
            val_loader = create_dataloader_label(1, configs.input_size, False, "sports_cars/val", "tree.p")
            color_model = ConvNet().to(device)
            color_model.to(torch.double)
            optimizer_full = torch.optim.Adam(color_model.parameters(), lr=configs.lr, weight_decay=.001)
            load(color_model, optimizer_full, model_name)
            VGGnet.to(torch.double)

            correct = 0
            n_guesses = 0
            color_model.eval()
            with torch.no_grad():
                for X, Weights, ii, label in tqdm(val_loader, leave=False):
                    X, Z = encode(X, Weights, ii, device)
                    del ii
                    del Weights
                    val_batch_Z = color_model(X)
                    val_batch_im = pred_to_rgb_vec(X, val_batch_Z, device, T=0.38)
                    hmm = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225])])
                    output = VGGnet(hmm(val_batch_im.to(torch.double)))
                    _, guess_catid = torch.topk(torch.nn.functional.softmax(output[0], dim=0), 1)
                    n_guesses += 1
                    if categories[guess_catid] == label[0]:
                        correct += 1
                    accuracies[acc_index] = correct / n_guesses
    for i in range(len(accuracies)):
        print(models_to_test[i] + ' accuracy: ' + str(accuracies[i]))


if __name__ == '__main__':
    configs = parse_configs()
    VGG_eval(configs)
