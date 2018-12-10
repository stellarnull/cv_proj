# reference: https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer

import argparse
import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import utils
from net import Net, Vgg16

class Msg_Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="parser for PyTorch-Style-Transfer")
        self.parser.add_argument("--content-img", type=str, required=True,
                                help="path to content image you want to stylize")
        self.parser.add_argument("--style-img", type=str, default="images/9styles/candy.jpg",
                                help="path to style-image")
        self.parser.add_argument("--output-img", type=str, default="output.jpg",
                                help="path for saving the output image")
        self.parser.add_argument("--model", type=str, required=True,
                                help="saved model to be used for stylizing the image")

    def parse(self):
        return self.parser.parse_args()


def main():
    args = Msg_Options().parse()
    evaluate(args)

def evaluate(args):
    content_img = utils.tensor_load_rgbimage(args.content_img, size=512, keep_asp=True)
    content_img = content_img.unsqueeze(0)
    style = utils.tensor_load_rgbimage(args.style_img, size=512)
    style = style.unsqueeze(0)    
    style = utils.preprocess_batch(style)


    model_dict = torch.load(args.model)
    model_dict_clone = model_dict.copy() # We can't mutate while iterating

    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]


    style_model = Net(128)
    style_model.load_state_dict(model_dict, False)

    if torch.cuda.is_available():
        style_model.cuda()
        content_img = content_img.cuda()
        style = style.cuda()

    style_v = Variable(style)

    content_img = Variable(utils.preprocess_batch(content_img))
    style_model.setTarget(style_v)

    output = style_model(content_img)
    #output = utils.color_match(output, style_v)
    utils.tensor_save_bgrimage(output.data[0], args.output_img, torch.cuda.is_available())


if __name__ == "__main__":
   main()
