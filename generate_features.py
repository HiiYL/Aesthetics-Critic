import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

cudnn.benchmark = True

import numpy as np
import os
from data_loader import get_loader 
from build_vocab import Vocabulary
from models_adversarial import EncoderCNN, G,D, InceptionNet
import pickle
import datetime
import torchvision.datasets as dset

from tensorboard_logger import configure, log_value

def train(save_path, args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    transform = transforms.Compose([
        #transforms.RandomCrop(args.crop_size),
        transforms.CenterCrop(args.crop_size),
        #transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    

    training_location = "data/{}/train".format(args.dataset)
    # Build data loader
    data_loader = dset.ImageFolder(training_location, transform)

    # Build the models
    encoder = EncoderCNN(args.embed_size, torch.load('data/net_model_epoch_10.pth').inception)

    if torch.cuda.is_available():
        encoder = encoder.cuda()


    # Train the Models
    total_step = len(data_loader)

    total_iterations = 0

    features_list = []
    for i, images in enumerate(data_loader):


        for p in netD.parameters():  # reset require_grad
            p.requires_grad = True   # they are set to False below in netG update
        
        # Set mini-batch dataset
        images = Variable(images, volatile=True)
        if torch.cuda.is_available():
            images = images.cuda()


        features = encoder(images)

        features_list.append(features)

        # Print log info
        if total_iterations % args.log_step == 0:
            print('Epoch [%d/%d], Step [%d/%d], G_Loss: %.4f, D_Loss: %5.4f'
                  %(epoch, args.num_epochs, i, total_step, 
                    G_loss.data[0], D_loss.data[0]))

        total_iterations += 1

        pickle.dump(features_list, open("features.pkl", "wb"))


                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=299 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--dataset', type=str, default='aesthetics' ,
                        help='dataset to use')
    parser.add_argument('--comments_path', type=str,
                        default='data/labels.h5',
                        help='path for train annotation json file')
    # parser.add_argument('--image_dir', type=str, default='./data/resized2014' ,
    #                     help='directory for resized images')
    # parser.add_argument('--comments_path', type=str,
    #                     default='./data/annotations/captions_train2014.json',
    #                     help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--tb_log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=10000,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('--pretrained', type=str)#, default='-2-20000.pkl')
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clip', type=float, default=1.0,help='gradient clipping')
    args = parser.parse_args()
    print(args)

    if not os.path.exists("logs"):
        os.mkdir("logs")

    if not os.path.exists(os.path.join("logs", args.dataset)):
        os.mkdir(os.path.join("logs", args.dataset))

    now = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
    save_path = os.path.join(os.path.join("logs", args.dataset), now)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    configure(save_path)
    train(save_path, args)