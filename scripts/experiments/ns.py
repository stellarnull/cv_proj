import os
import sys
import argparse
import os
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

class Ns_Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="parser for PyTorch-Style-Transfer")
        self.parser.add_argument("--content-img", type=str, default="images/content/columbia.jpg",
                                help="path to content image you want to stylize")
        self.parser.add_argument("--style-img", type=str, default="images/9styles/candy.jpg",
                                help="path to style-image")
        self.parser.add_argument("--output-img", type=str, default="output.jpg",
                                help="path for saving the output image")
        self.parser.add_argument("--ratio", type=float, default=1.0,
                                help="ratio of content loss / style loss")  

    def parse(self):
        return self.parser.parse_args()


def main():
    args = Ns_Options().parse()
    ns_run(args)

def ns_run(args):
    # load files
    content_img = utils.tensor_load_rgbimage(args.content_img, size=512, keep_asp=True)
    content_img = content_img.unsqueeze(0)
    content_img = Variable(utils.preprocess_batch(content_img), requires_grad=False)
    content_img = utils.subtract_imagenet_mean_batch(content_img)
    style_img = utils.tensor_load_rgbimage(args.style_img, size=512)
    style_img = style_img.unsqueeze(0)    
    style_img = Variable(utils.preprocess_batch(style_img), requires_grad=False)
    style_img = utils.subtract_imagenet_mean_batch(style_img)

    # load the pre-trained vgg network
    vgg = Vgg16()
    utils.init_vgg16("models/")
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, "vgg16.weight")))
    if torch.cuda.is_available():
        content_img = content_img.cuda()
        style_img = style_img.cuda()
        vgg.cuda()
    features_content = vgg(content_img)
    f_xc_c = Variable(features_content[1].data, requires_grad=False)
    features_style = vgg(style_img)
    gram_style = [utils.gram_matrix(y) for y in features_style]
    # gram_style = [utils.gram_matrix(features_style[3])]

    output = Variable(content_img.data, requires_grad=True)
    ns_runr = Adam([output], lr=1e1)
    mse_loss = torch.nn.MSELoss()

    # training the images
    bar = trange(500)
    for _iter in bar:
        utils.imagenet_clamp_batch(output, 0, 255)
        ns_runr.zero_grad()
        features_y = vgg(output)
        content_loss = mse_loss(features_y[1], f_xc_c)

        style_loss = 0.
        for m in range(len(features_y)):
            gram_y = utils.gram_matrix(features_y[m])
            gram_s = Variable(gram_style[m].data, requires_grad=False)
            style_loss += mse_loss(gram_y, gram_s)
        # gram_y = utils.gram_matrix(features_y[3])
        # gram_s = Variable(gram_style[0].data, requires_grad=False)
        # style_loss += args.style_weight * mse_loss(gram_y, gram_s)

        alpha = args.ratio / (args.ratio + 1)
        beta = 1 / (args.ratio + 1)
        total_loss = alpha * content_loss + beta * style_loss
        total_loss.backward()
        ns_runr.step()
        # bar.set_description(total_loss.data.cpu().numpy()[0])
    # save the image to output file   
    output = utils.add_imagenet_mean_batch(output)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)

if __name__ == "__main__":
   main()
