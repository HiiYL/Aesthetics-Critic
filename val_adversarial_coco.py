import argparse


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torchvision import transforms
from torchvision import models

cudnn.benchmark = True

import numpy as np
import os
from data_loader_coco import get_loader 
from build_vocab import Vocabulary
from models_adversarial import EncoderCNN, G,D, InceptionNet
import pickle
import datetime

import json

from tensorboard_logger import configure, log_value


## Ms COCO eval code imports
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import os

torch.manual_seed(123)

def train(save_path, args):
    ## set up file names and pathes
    dataDir='data/coco'
    logDir=save_path
    dataType='val2014'
    algName = 'fakecap'
    annFile='%s/captions_%s.json'%(dataDir,dataType)
    subtypes=['results', 'evalImgs', 'eval']
    [resFile, evalImgsFile, evalFile]= \
    ['%s/captions_%s_%s_%s.json'%(logDir,dataType,algName,subtype) for subtype in subtypes]
    ###

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Scale(299),
        transforms.RandomCrop(299),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    val_data_loader = get_loader("val", vocab, 
                             transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size,models.inception_v3(pretrained=True))
    netG    = G(args.embed_size, args.hidden_size, vocab, args.num_layers)
    if args.pretrained:
        print("loading pretrained model...")
        netG.load_state_dict(torch.load(args.pretrained))

    real_label = 1
    fake_label = 0
    state      = (Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)),
        Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)))
    criterion  = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        netG.cuda()
        # netD.cuda()
        state = [s.cuda() for s in state]
        criterion = nn.CrossEntropyLoss()

    print('validating...')
    netG.eval()
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad=False
    for parameter in netG.parameters():
        parameter.requires_grad=False

    val_json = []
    total_val_step = len(val_data_loader)
    total_iterations = 0
    for i, (images, captions, lengths, img_id) in enumerate(val_data_loader):
        if i > 100:
            break
        images = Variable(images, volatile=True)
        captions = Variable(captions, volatile=True)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        targets, batch_sizes = pack_padded_sequence(captions, lengths, batch_first=True)

        # Forward, Backward and Optimize
        features = encoder(images)

        sampled_ids_list, terminated_confidences = netG.beamSearch(features, state, n=10, diverse_gamma=0.5)
        sampled_ids_list = np.array([ sampled_ids.cpu().data.numpy() for sampled_ids in sampled_ids_list ])
        
        # Decode word_ids to words
        sorted_idx = np.argsort(terminated_confidences)
        sampled_ids_sorted = sorted(zip(terminated_confidences,sampled_ids_list), reverse=True)

        _, sampled_ids = sampled_ids_sorted[0]

        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word == '<end>':
                break
            if word != '<start>' and word != '<pad>':
                sampled_caption.append(word)

        sentence = ' '.join(sampled_caption)
        print(sentence)
        item_json = {"image_id": img_id[0], "caption": sentence}
        val_json.append(item_json)
        
        # print()
        # for sampled_with_confidence in sampled_ids_sorted:
        #     confidence, sample  = sampled_with_confidence
        #     sampled_caption = []
        #     for word_id in sample:
        #         word = vocab.idx2word[word_id]
        #         sampled_caption.append(word)
        #         if word == '<end>':
        #             break
        #     sentence = ' '.join(sampled_caption)
            
        #     # Print out image and generated caption.
        #     print ("{} - {:.4f} ".format(sentence, confidence))
        # print()


        # # print(sampled_ids)
        # # sampled_ids = torch.max(outputs,1)[1].squeeze()
        # # sampled_ids = pad_packed_sequence([sampled_ids, batch_sizes], batch_first=True)[0]
        # sampled_ids = sampled_ids.cpu().data.numpy()
        # loss        = criterion(outputs, targets)

        # for j, comment in enumerate(sampled_ids):
        #     sampled_caption = []
        #     for word_id in comment:
        #         word = vocab.idx2word[word_id]
        #         if word == '<end>':
        #             break
        #         if word != '<start>' and word != '<pad>':
        #             sampled_caption.append(word)

        #     sentence = ' '.join(sampled_caption)
        #     print(sentence)
        #     item_json = {"image_id": img_id[j], "caption": sentence}
        #     val_json.append(item_json)

        # Print log info
        if i % args.log_step == 0:
           print('[%d/%d] - Running model on validation set....'
                 %(i, total_val_step))

    path, file = os.path.split(resFile)

    export_filename = '{}/{}_{}'.format(path, total_iterations, file)
    print("Exporting to... {}".format(export_filename))
    with open(export_filename, 'w') as outfile:
        json.dump(val_json, outfile)

    print("calculating metrics...")
    coco = COCO(annFile)
    cocoRes = coco.loadRes(export_filename)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # for metric, score in cocoEval.eval.items():
    #     log_value(metric, score, total_iterations)
    #     print '%s: %.3f'%(metric, score)



            


                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=299 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab-py2.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--dataset', type=str, default='coco' ,
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
    parser.add_argument('--tb_log_step', type=int , default=100,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=10000,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of gru hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in gru')
    parser.add_argument('--pretrained', type=str, default='logs/coco/03052017180131/decoder-1-20000.pkl')#, default='-2-20000.pkl')
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
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

    # configure(save_path)
    train(save_path, args)