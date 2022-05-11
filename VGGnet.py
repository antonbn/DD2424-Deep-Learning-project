import torch
from VGG_dataloader import create_dataloader
from tqdm import tqdm
from config import parse_configs
import os

def VGG_download(path):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
    torch.save(model, path)
    return model

def VGG_load(path):
    return torch.load(path)

def VGG_eval(configs):
    path = 'pretrained_VGG.pt'
    if not os.path.exists('pretrained_VGG.pt'):
        model = VGG_download(path)
    else:
        model = VGG_load(path)

    model.eval()

    val_loader_VGG = create_dataloader(1, configs.input_size, True, "not val")

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
        if categories[guess_catid] in y[0]:
            correct += 1
        #print('Predicted category: ' + str(categories[guess_catid]) + '\n', 'True category: ' + str(y[0]) + '\n', 'Confidence: ' + str(guess_prob.item()) + '\n')
    print('Total accuracy: ' + str(correct/n_guesses))


if __name__ == '__main__':
    configs = parse_configs()
    VGG_eval(configs)