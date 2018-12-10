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

def main():
    train()

def train():
    check_paths()
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    transform = transforms.Compose([transforms.Scale(256),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder("dataset/", transform)
    train_loader = DataLoader(train_dataset, batch_size=4, **kwargs)

    style_model = Net(128)
    print(style_model)
    optimizer = Adam(style_model.parameters(), 1e-3)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16()
    utils.init_vgg16("models/")
    vgg.load_state_dict(torch.load(os.path.join("models/", "vgg16.weight")))

    if torch.cuda.is_available():
        style_model.cuda()
        vgg.cuda()

    style_loader = utils.StyleLoader("images/9styles/", 512)

    tbar = trange(2)
    for e in tbar:
        style_model.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(utils.preprocess_batch(x))
            if torch.cuda.is_available():
                x = x.cuda()

            style_v = style_loader.get(batch_id)
            style_model.setTarget(style_v)

            style_v = utils.subtract_imagenet_mean_batch(style_v)
            features_style = vgg(style_v)
            gram_style = [utils.gram_matrix(y) for y in features_style]

            y = style_model(x)
            xc = Variable(x.data.clone())

            y = utils.subtract_imagenet_mean_batch(y)
            xc = utils.subtract_imagenet_mean_batch(xc)

            features_y = vgg(y)
            features_xc = vgg(xc)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            content_loss = mse_loss(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_y = utils.gram_matrix(features_y[m])
                gram_s = Variable(gram_style[m].data.cuda(), requires_grad=False).repeat(4, 1, 1)
                style_loss += 5.0 * mse_loss(gram_y, gram_s[:n_batch, :, :])

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]

            if (batch_id + 1) % 500 == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                agg_content_loss / (batch_id + 1),
                                agg_style_loss / (batch_id + 1),
                                (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                tbar.set_description(mesg)

            
            if (batch_id + 1) % (4 * 500) == 0:
                # save model
                style_model.eval()
                style_model.cpu()
                save_model_filename = "Epoch_" + str(e) + "iters_" + str(count) + "_" + \
                    str(time.ctime()).replace(' ', '_') + "_" + str(
                    1.0) + "_" + str(5.0) + ".model"
                save_model_path = os.path.join("models/", save_model_filename)
                torch.save(style_model.state_dict(), save_model_path)
                style_model.train()
                style_model.cuda()
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

    # save model
    style_model.eval()
    style_model.cpu()
    save_model_filename = "Final_epoch_" + str(2) + "_" + \
        str(time.ctime()).replace(' ', '_') + "_" + str(
        1.0) + "_" + str(5.0) + ".model"
    save_model_path = os.path.join("models/", save_model_filename)
    torch.save(style_model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def check_paths():
    try:
        if not os.path.exists("models/"):
            os.makedirs("models/")
    except OSError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
   main()
