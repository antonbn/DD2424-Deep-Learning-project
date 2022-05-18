import numpy as np
import torch
from VGG_dataloader import create_dataloader
from tqdm import tqdm
from config import parse_configs
from annealed_mean import pred_to_rgb_vec
from matplotlib import pyplot as plt
import os

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
        model = VGG_download(path)
    else:
        model = VGG_load(path)

    model.eval()

    val_loader_VGG = create_dataloader(1, configs.input_size, True, "val", 'normalize')

    # Have to download the imagenet_classes.txt to the VM if it isn't there
    with open('..\ImageNet\LOC_synset_mapping.txt', "r") as f:
        categories = [s.strip() for s in f.readlines()]
    categories = [x.split(" ")[0] for x in categories]

    correct = 0
    n_guesses = 0
    for x,y in tqdm(val_loader_VGG, total=len(val_loader_VGG)):
        with torch.no_grad():
            output = model(x)
        _, guess_catid = torch.topk(torch.nn.functional.softmax(output[0], dim=0), 1)
        n_guesses += 1
        #print(categories[guess_catid])
        #plt.imshow(x[0].moveaxis(0,-1))
        #plt.show()
        if categories[guess_catid] in y[0]:
            correct += 1
        #print('Predicted category: ' + str(categories[guess_catid]) + '\n', 'True category: ' + str(y[0]) + '\n', 'Confidence: ' + str(guess_prob.item()) + '\n')
    print('Total accuracy: ' + str(correct/n_guesses))


if __name__ == '__main__':
    configs = parse_configs()
    VGG_eval(configs)