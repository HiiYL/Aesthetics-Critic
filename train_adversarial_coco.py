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

##







def run(save_path, args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Scale((299,299)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    train_data_loader = get_loader("train", vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    val_data_loader = get_loader("val", vocab, 
                             transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size,models.inception_v3(pretrained=True))
    netG = G(args.embed_size, args.hidden_size, vocab, args.num_layers)
    netD = D(args.embed_size, args.hidden_size, vocab, args.num_layers)
    if args.pretrained:
        netG.load_state_dict(torch.load(args.pretrained))

    label_real, label_fake = torch.FloatTensor(args.batch_size), torch.FloatTensor(args.batch_size)
    real_label = 1
    fake_label = 0

    criterion = nn.CrossEntropyLoss()
    criterion_bce = nn.BCELoss()
    
    state = (Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)),
     Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)))

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        netG.cuda()
        netD.cuda()
        state = [s.cuda() for s in state]
        label_real, label_fake = label_real.cuda(), label_fake.cuda()
        criterion_bce = criterion_bce.cuda()
        criterion = criterion.cuda()

    label_real, label_fake = Variable(label_real), Variable(label_fake)

    # optimizerE = torch.optim.Adam(encoder.parameters(), lr= 0.1 *args.learning_rate)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.learning_rate)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.learning_rate)

    # Train the Models
    total_step = len(train_data_loader)

    total_iterations = 0

    for epoch in range(args.num_epochs):
        # if epoch % 8 == 0:
        #     lr = args.learning_rate * (0.5 ** (epoch // 8))
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        for i, (images, captions, lengths, img_id) in enumerate(train_data_loader):
            for p in netD.parameters():  # reset require_grad
                p.requires_grad = True   # they are set to False below in netG update
            
            # Set mini-batch dataset
            images = Variable(images)
            captions = Variable(captions)
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()

            targets, batch_sizes = pack_padded_sequence(captions, lengths, batch_first=True)

            netG.zero_grad()
            netD.zero_grad()
            features = encoder(images).detach()
            out, hidden, embeddings   = netG(features, captions, lengths, state, teacher_forced=True)
            outputs_free, hidden_free = netG(features, captions, lengths, state, teacher_forced=False)

            ## Discriminator Step
            output_real      = netD(targets,lengths,batch_sizes, label=True)
            label_real.data.resize_(output_real.size()).fill_(real_label)
            D_loss_real      = criterion_bce(output_real, label_real)
            
            output_fake      = netD(outputs_free.detach(),lengths,batch_sizes, label=False)
            label_fake.data.resize_(output_fake.size()).fill_(fake_label)
            D_loss_fake      = criterion_bce(output_fake, label_fake)
            
            total = len(label_real)
            correct_real = (output_real.data >= 0.5).sum() / total
            correct_fake = (output_fake.data < 0.5).sum() / total
            D_accuracy = 0.5 * ( correct_real + correct_fake )
            D_loss = D_loss_real + D_loss_fake

            if not D_accuracy > 0.99:
                D_loss.backward()
                optimizerD.step()

            ## Generator Step
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()

            output_free = netD(outputs_free,lengths,batch_sizes, label=False)
            gan_loss = criterion_bce(output_free, label_real)

            mle_loss = criterion(out, targets)
            if D_accuracy > 0.75:
                G_loss = mle_loss + gan_loss #+ gan_loss_real
            else:
                G_loss = mle_loss
            G_loss.backward()
            optimizerG.step()

            # Print log info
            if total_iterations % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], G_Loss: %.4f, D_Loss: %5.4f, Perplexity: %5.4f, D_Accuracy: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        G_loss.data[0], D_loss.data[0], np.exp(mle_loss.data[0]), D_accuracy)) 

            if total_iterations % 100 == 0:
                print("")
                sampled_ids_free = torch.max(outputs_free,1)[1].squeeze()
                sampled_ids_free = pad_packed_sequence([sampled_ids_free, batch_sizes], batch_first=True)[0]
                sampled_ids_free = sampled_ids_free.cpu().data.numpy()

                sampled_ids_forced = torch.max(out,1)[1].squeeze()
                sampled_ids_forced = pad_packed_sequence([sampled_ids_forced, batch_sizes], batch_first=True)[0]
                sampled_ids_forced = sampled_ids_forced.cpu().data.numpy()

                for i, comment in enumerate(sampled_ids_free):
                    if i > 1:
                        break
                    sampled_caption = []
                    sample = captions.cpu().data.numpy()
                    for word_id in sample[i]:
                        word = vocab.idx2word[word_id]
                        sampled_caption.append(word)
                        if word == '<end>':
                            break
                    sentence = "[G]" + ' '.join(sampled_caption)
                    print(sentence)

                    sampled_caption = []
                    for word_id in sampled_ids_forced[i]:
                        word = vocab.idx2word[word_id]
                        sampled_caption.append(word)
                        if word == '<end>':
                            break
                    sentence = "[T]" + ' '.join(sampled_caption)
                    print(sentence)

                    sampled_caption = []
                    for word_id in comment:
                        word = vocab.idx2word[word_id]
                        sampled_caption.append(word)
                        if word == '<end>':
                            break
                    sentence = "[F]" + ' '.join(sampled_caption)
                    print(sentence)

                    print("")
            # Save the model
            if (total_iterations+1) % args.save_step == 0:
                # torch.save(encoder.state_dict(), 
                #            os.path.join(save_path, 
                #                         'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(netG.state_dict(), 
                           os.path.join(save_path, 
                                        'netG-%d-%d.pkl' %(epoch+1, i+1)))
                # torch.save(netD.state_dict(), 
                #            os.path.join(save_path, 
                #                         'netD-%d-%d.pkl' %(epoch+1, i+1)))


            if total_iterations % args.tb_log_step == 0:
                log_value('Loss', mle_loss.data[0], total_iterations)
                log_value('Perplexity', np.exp(mle_loss.data[0]), total_iterations)

            if total_iterations % 10000 == 0:
                validate(encoder, netG, val_data_loader, state, criterion, vocab, total_iterations)

            total_iterations += 1



def validate(encoder, netG, val_data_loader, state, criterion, vocab, total_iterations):
    ### MS COCO Eval code prepation
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
        outputs, _ = netG(features, captions, lengths, state, teacher_forced=False)
        sampled_ids = torch.max(outputs,1)[1].squeeze()
        sampled_ids = pad_packed_sequence([sampled_ids, batch_sizes], batch_first=True)[0]
        sampled_ids = sampled_ids.cpu().data.numpy()
        loss        = criterion(outputs, targets)

        for j, comment in enumerate(sampled_ids):
            sampled_caption = []
            for word_id in comment:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                if word != '<start>' and word != '<pad>':
                    sampled_caption.append(word)

            item_json = {"image_id": img_id[j], "caption":' '.join(sampled_caption) }
            val_json.append(item_json)

        # Print log info
        if i % args.log_step == 0:
           print('[%d/%d] - Running model on validation set.... | Loss: %.4f | Perplexity: %5.4f'
                 %(i, total_val_step, 
                   loss.data[0], np.exp(loss.data[0])))


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

    configure(save_path)
    run(save_path, args)