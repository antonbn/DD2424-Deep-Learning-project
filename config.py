import argparse
from easydict import EasyDict as edict


def parse_configs():
    parser = argparse.ArgumentParser()
    ####################################################################
    ##############     Model configs            ###################
    ####################################################################
    parser.add_argument('--input_size', type=int, default=224, metavar='INPSIZE',
                        help='The size of the input image')

    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--name', type=str, default='model', metavar='Name',
                        help='model name')
    parser.add_argument('--batch_size', type=int, default=32)

    ####################################################################
    ##############     Training strategy            ###################
    ####################################################################

    parser.add_argument('--num_epochs', type=int, default=20, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='initial learning rate')

    configs = edict(vars(parser.parse_args()))

    return configs