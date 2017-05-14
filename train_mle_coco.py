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
from models_spatial import EncoderCNN, G_Spatial
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

def run(save_path, args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    train_transform = transforms.Compose([
        transforms.Scale((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    test_transform = transforms.Compose([
        transforms.Scale((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    train_data_loader = get_loader("train", vocab, 
                             train_transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    val_data_loader = get_loader("val", vocab, 
                             test_transform, 1,
                             shuffle=False, num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size,models.inception_v3(pretrained=True), requires_grad=False)
    netG = G_Spatial(args.embed_size, args.hidden_size, vocab, args.num_layers)

    if args.netG:
        print("[!]loading pretrained netG....")
        netG.load_state_dict(torch.load(args.netG))
        print("Done!")

    if args.encoder:
        encoder.load_state_dict(torch.load(args.encoder))

    criterion = nn.CrossEntropyLoss()
    
    state = (Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)),
     Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)))
    val_state = (Variable(torch.zeros(args.num_layers, 1, args.hidden_size)),
     Variable(torch.zeros(args.num_layers, 1, args.hidden_size)))

    y_onehot = torch.FloatTensor(args.batch_size, 20,len(vocab))

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        netG.cuda()
        #netD.cuda()
        state = [s.cuda() for s in state]
        val_state = [s.cuda() for s in val_state]
        criterion = criterion.cuda()
        y_onehot = y_onehot.cuda()

    params = [
                {'params': netG.parameters()},
                #{'params': encoder.parameters(), 'lr': 0.1 * args.learning_rate}
                #{'params': encoder.fc.parameters()}
                
            ]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate,betas=(0.8, 0.999))

    # Train the Models
    total_step = len(train_data_loader)
    total_iterations = 0

    for epoch in range(args.num_epochs):
        # if epoch % 8 == 0:
        #     lr = args.learning_rate * (0.5 ** (epoch // 8))
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        for i, (images, captions, lengths, img_id) in enumerate(train_data_loader):
            # Set mini-batch dataset
            images = Variable(images)
            captions = Variable(captions)
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()

            targets, batch_sizes = pack_padded_sequence(captions, lengths, batch_first=True)

            y_onehot.resize_(captions.size(0),captions.size(1),len(vocab))
            y_onehot.zero_()
            y_onehot.scatter_(2,captions.data.unsqueeze(2),1)
            y_v = Variable(y_onehot)

            netG.zero_grad()
            inputs = Variable(images.data, volatile=True)
            features = Variable(encoder(inputs).data)
            features_g, features_l = netG.encode_fc(features)
            #features_g, features_l = features_g.detach(), features_l.detach()
            out = netG((features_g, features_l), y_v, lengths, state, teacher_forced=True)

            mle_loss = criterion(out, targets)
            mle_loss.backward()
            torch.nn.utils.clip_grad_norm(netG.parameters(), args.clip)
            optimizer.step()

            # Print log info
            if total_iterations % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d] Loss: %5.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step,  mle_loss.data[0], np.exp(mle_loss.data[0]))) 

            # if total_iterations % 100 == 0:
            #     print("")
            #     outputs_free,_  = netG((features_g, features_l), y_v, lengths, state, teacher_forced=False)

            #     sampled_ids_free = torch.max(outputs_free,1)[1].squeeze()
            #     sampled_ids_free = pad_packed_sequence([sampled_ids_free, batch_sizes], batch_first=True)[0]
            #     sampled_ids_free = sampled_ids_free.cpu().data.numpy()

            #     def word_idx_to_sentence(sample):
            #         sampled_caption = []
            #         for word_id in sample:
            #             word = vocab.idx2word[word_id]
            #             sampled_caption.append(word)
            #             if word == '<end>':
            #                 break
            #         return ' '.join(sampled_caption)

            #     groundtruth_caption = captions.cpu().data.numpy()
            #     sampled_captions = ""
            #     for i, comment in enumerate(sampled_ids_free):
            #         if i > 1:
            #             break

            #         sampled_caption   = word_idx_to_sentence(groundtruth_caption[i])
            #         sampled_captions += "[G]{} \n".format(sampled_caption)

            #         sampled_caption =  word_idx_to_sentence(comment)
            #         sampled_captions += "[F]{} \n".format(sampled_caption)

            #     print(sampled_captions)

            # Save the model
            if (total_iterations+1) % args.save_step == 0:
                torch.save(netG.state_dict(), 
                           os.path.join(save_path, 
                                        'netG-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(save_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))


            if total_iterations % args.tb_log_step == 0:
                log_value('Loss', mle_loss.data[0], total_iterations)
                log_value('Perplexity', np.exp(mle_loss.data[0]), total_iterations)

            if (total_iterations+1) % args.save_step == 0:
                validate(encoder, netG, val_data_loader, val_state, criterion, vocab, total_iterations)

            total_iterations += 1



def validate(encoder, netG, val_data_loader, state, criterion, vocab, total_iterations):
    ### MS COCO Eval code prepation
    ## set up file names and pathes
    dataDir='data/coco'
    logDir=save_path
    dataType='val2014'
    algName = 'fakecap'
    annFile='%s/captions_%s_karpathy_split.json'%(dataDir,dataType)
    subtypes=['results', 'evalImgs', 'eval']
    [resFile, evalImgsFile, evalFile]= \
    ['%s/captions_%s_%s_%s.json'%(logDir,dataType,algName,subtype) for subtype in subtypes]
    ###


    print('validating...')
    netG.eval()
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad=False
    for parameter in netG.parameters():
        parameter.requires_grad=False



    val_json = []
    total_val_step = len(val_data_loader)
    for i, (images, captions, lengths, img_id) in enumerate(val_data_loader):
        images = Variable(images, volatile=True)
        captions = Variable(captions, volatile=True)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        targets, batch_sizes = pack_padded_sequence(captions, lengths, batch_first=True)

        # Forward, Backward and Optimize
        features = encoder(images)
        features_g, features_l = netG.encode_fc(features)

        sampled_ids_list, terminated_confidences = netG.beamSearch((features_g, features_l), state, n=3, diverse_gamma=0.0)
        sampled_ids_list = np.array([ sampled_ids.cpu().data.numpy() for sampled_ids in sampled_ids_list ])

        max_index = np.argmin(terminated_confidences) ## Should this be min?
        #print(max_index)
        sampled_ids = sampled_ids_list[max_index]

        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word == '<end>' or word == '<pad>':
                break
            if word != '<start>':
                sampled_caption.append(word)

        sentence = ' '.join(sampled_caption)
        item_json = {"image_id": img_id[0], "caption": sentence}
        val_json.append(item_json)
        # outputs, _ = netG(features, captions, lengths, state, teacher_forced=False)
        # sampled_ids = torch.max(outputs,1)[1].squeeze()
        # sampled_ids = pad_packed_sequence([sampled_ids, batch_sizes], batch_first=True)[0]
        # sampled_ids = sampled_ids.cpu().data.numpy()
        # loss        = criterion(outputs, targets)

        # for j, comment in enumerate(sampled_ids):
        #     sampled_caption = []
        #     for word_id in comment:
        #         word = vocab.idx2word[word_id]
        #         if word == '<end>':
        #             break
        #         elif word == '<start>' and word == '<pad>':
        #             continue
        #         else:
        #             sampled_caption.append(word)

        #     item_json = {"image_id": img_id[j], "caption":' '.join(sampled_caption) }
        #     val_json.append(item_json)

        # # Print log info
        if i % args.log_step * 10 == 0:
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

    for metric, score in cocoEval.eval.items():
        log_value(metric, score, total_iterations)
        print '%s: %.3f'%(metric, score)

    netG.train()
    encoder.train()
    for parameter in encoder.parameters():
        parameter.requires_grad=True
    for parameter in netG.parameters():
        parameter.requires_grad=True
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=299 ,
                        help='image size to use')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--dataset', type=str, default='coco' ,
                        help='dataset to use')
    parser.add_argument('--comments_path', type=str,
                        default='data/labels.h5',
                        help='path for train annotation json file')
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
    parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
    parser.add_argument('--netG', type=str, default="logs/coco/14052017174557/netG-1-10000.pkl")
    parser.add_argument('--encoder', type=str, default="logs/coco/14052017174557/encoder-1-10000.pkl")
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
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
    run(save_path, args)