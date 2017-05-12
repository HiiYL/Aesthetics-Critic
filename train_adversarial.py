import argparse


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, grad
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torchvision import transforms
from torchvision import models

cudnn.benchmark = True

import numpy as np
import os
from data_loader import get_loader 
from build_vocab import Vocabulary
from models_adversarial import EncoderCNN, G, D, D_Attention, InceptionNet
import pickle
import datetime

import json

from tensorboard_logger import configure, log_value



# ## Ms COCO eval code imports
# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
# import matplotlib.pyplot as plt
# import skimage.io as io

# import json
# from json import encoder
# encoder.FLOAT_REPR = lambda o: format(o, '.3f')
# import os

def run(save_path, args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Scale(299),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    training_location = "data/{}/train".format(args.dataset)
    train_data_loader = get_loader(training_location, args.comments_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models
    encoder = EncoderCNN(args.embed_size,torch.load('data/net_model_epoch_10.pth').inception)#models.inception_v3(pretrained=True), requires_grad=False)
    netG = G(args.embed_size, args.hidden_size, vocab, args.num_layers)
    netD = D_Attention(args.embed_size, args.hidden_size, vocab, args.num_layers, use_log=True)
    if args.netG:
        print("[!]loading pretrained model for generator....")
        netG.load_state_dict(torch.load(args.netG))
        print("Done!")

    if args.netE:
        print("[!]loading pretrained model for encoder....")
        encoder.load_state_dict(torch.load(args.netE))
        print("Done!")

    label_real, label_fake = torch.FloatTensor(args.batch_size), torch.FloatTensor(args.batch_size)

    one = torch.FloatTensor([1])
    mone = one * -1

    criterion = nn.CrossEntropyLoss()
    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    
    state = (Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)),
     Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)))
    val_state = (Variable(torch.zeros(args.num_layers, 1, args.hidden_size)),
     Variable(torch.zeros(args.num_layers, 1, args.hidden_size)))
    alpha = torch.FloatTensor(args.batch_size,1,1,1).uniform_(0,1)
    y_onehot = torch.FloatTensor(args.batch_size, len(vocab))

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        netG.cuda()
        netD.cuda()
        state = [s.cuda() for s in state]
        val_state = [s.cuda() for s in val_state]
        label_real, label_fake = label_real.cuda(), label_fake.cuda()
        criterion_bce = criterion_bce.cuda()
        criterion = criterion.cuda()
        criterion_mse = criterion_mse.cuda()
        one, mone = one.cuda(), mone.cuda()
        alpha = alpha.cuda()
        y_onehot = y_onehot.cuda()


    label_real, label_fake = Variable(label_real), Variable(label_fake)
    real_label = 1
    fake_label = 0

    #optimizerE = torch.optim.Adam(encoder.parameters(), lr= 0.1 * args.learning_rate)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.learning_rate)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.learning_rate)

    # Train the Models
    total_step = len(train_data_loader)
    total_iterations = 0
    gen_iterations   = 0
    current_iter = 0
    running_accuracy_average = 0
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(train_data_loader):
            # Set mini-batch dataset
            images = Variable(images)
            captions = Variable(captions, lengths)
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()

            for p in netD.parameters():  # reset require_grad
                p.requires_grad = True   # they are set to False below in netG update
            

            targets, batch_sizes = pack_padded_sequence(captions, lengths, batch_first=True)

            netG.zero_grad()
            netD.zero_grad()

            y_onehot.resize_(targets.size(0),len(vocab))
            y_onehot.zero_()
            y_onehot.scatter_(1,targets.data.unsqueeze(1),1)
            y_v = Variable(y_onehot)

            features = encoder(images).detach()
            outputs_free, _ = netG(features, captions, lengths, state, teacher_forced=False)

            ## Discriminator Step
            output_real, D_hidden_real = netD(features.detach(), y_v, lengths, state, batch_sizes)
            label_real.data.resize_(output_real.size()).fill_(real_label)
            D_loss_real = criterion_bce(output_real, label_real)

            output_fake, _  = netD(features.detach(), outputs_free.detach(), lengths, state, batch_sizes)
            label_fake.data.resize_(output_fake.size()).fill_(fake_label)
            D_loss_fake = criterion_bce(output_fake, label_fake)

            total = len(label_real)
            correct_fake = (output_fake.data < 0.5).sum() / total
            D_accuracy = correct_fake

            D_loss = D_loss_real + D_loss_fake
            D_loss.backward()
            optimizerD.step()
            
            current_iter += 1
            total_iterations += 1
            if current_iter > args.Diters:
                current_iter = 0
                gen_iterations += 1
                ## Generator Step
                for p in netD.parameters():
                    p.requires_grad = False # to avoid computation
                netG.zero_grad()
                netD.zero_grad()
                output_fake, D_hidden_fake = netD(features, outputs_free,lengths, state, batch_sizes)
                G_loss = criterion_bce(output_fake, label_real)
                G_loss.backward()
                #mle_loss.backward()
                optimizerG.step()

                running_accuracy_average += D_accuracy
                # Print log info
                #if total_iterations % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], G_Loss: %.4f, D_Loss: %5.4f, D_Accuracy: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        G_loss.data[0], D_loss.data[0], running_accuracy_average/args.log_step)) 
                running_accuracy_average = 0

            if total_iterations % 100 == 0:
                print("")
                sampled_ids_free = torch.max(outputs_free,1)[1].squeeze()
                sampled_ids_free = pad_packed_sequence([sampled_ids_free, batch_sizes], batch_first=True)[0]
                sampled_ids_free = sampled_ids_free.cpu().data.numpy()

                def word_idx_to_sentence(sample):
                    sampled_caption = []
                    for word_id in sample:
                        word = vocab.idx2word[word_id]
                        sampled_caption.append(word)
                        if word == '<end>':
                            break
                    return ' '.join(sampled_caption)

                groundtruth_caption = captions.cpu().data.numpy()
                sampled_captions = ""
                for i, comment in enumerate(sampled_ids_free):
                    if i > 1:
                        break

                    sampled_caption   = word_idx_to_sentence(groundtruth_caption[i])
                    sampled_captions += "[G]{} \n".format(sampled_caption)

                    sampled_caption =  word_idx_to_sentence(comment)
                    sampled_captions += "[F]{} \n".format(sampled_caption)

                print(sampled_captions)

            # Save the model
            if (total_iterations) % args.save_step == 0:
                torch.save(netG.state_dict(), 
                           os.path.join(save_path, 
                                        'netG-%d.pkl' %(gen_iterations)))


            if (total_iterations) % args.tb_log_step == 0:
                log_value('D_Loss', D_loss.data[0], gen_iterations)
                log_value('G_Loss', G_loss.data[0], gen_iterations)




            # if (total_iterations) % 30000 == 0:
            #     validate(encoder, netG, val_data_loader, val_state, criterion, vocab, total_iterations)

            



# def validate(encoder, netG, val_data_loader, state, criterion, vocab, total_iterations):
#     ### MS COCO Eval code prepation
#     ## set up file names and pathes
#     dataDir='data/coco'
#     logDir=save_path
#     dataType='val2014'
#     algName = 'fakecap'
#     annFile='%s/captions_%s_karpathy_split.json'%(dataDir,dataType)
#     subtypes=['results', 'evalImgs', 'eval']
#     [resFile, evalImgsFile, evalFile]= \
#     ['%s/captions_%s_%s_%s.json'%(logDir,dataType,algName,subtype) for subtype in subtypes]
#     ###


#     print('validating...')
#     netG.eval()
#     encoder.eval()

#     for parameter in encoder.parameters():
#         parameter.requires_grad=False
#     for parameter in netG.parameters():
#         parameter.requires_grad=False



#     val_json = []
#     total_val_step = len(val_data_loader)
#     for i, (images, captions, lengths, img_id) in enumerate(val_data_loader):
#         images = Variable(images, volatile=True)
#         captions = Variable(captions, volatile=True)
#         if torch.cuda.is_available():
#             images = images.cuda()
#             captions = captions.cuda()
#         targets, batch_sizes = pack_padded_sequence(captions, lengths, batch_first=True)

#         # Forward, Backward and Optimize
#         features = encoder(images)

#         sampled_ids_list, terminated_confidences = netG.beamSearch(features, state, n=10, diverse_gamma=0.5)
#         sampled_ids_list = np.array([ sampled_ids.cpu().data.numpy() for sampled_ids in sampled_ids_list ])
        
#         # Decode word_ids to words
#         sorted_idx = np.argsort(terminated_confidences)
#         sampled_ids_sorted = sorted(zip(terminated_confidences,sampled_ids_list), reverse=True)

#         _, sampled_ids = sampled_ids_sorted[0]

#         sampled_caption = []
#         for word_id in sampled_ids:
#             word = vocab.idx2word[word_id]
#             if word == '<end>':
#                 break
#             if word != '<start>' and word != '<pad>':
#                 sampled_caption.append(word)

#         sentence = ' '.join(sampled_caption)
#         item_json = {"image_id": img_id[0], "caption": sentence}
#         val_json.append(item_json)
#         # outputs, _ = netG(features, captions, lengths, state, teacher_forced=False)
#         # sampled_ids = torch.max(outputs,1)[1].squeeze()
#         # sampled_ids = pad_packed_sequence([sampled_ids, batch_sizes], batch_first=True)[0]
#         # sampled_ids = sampled_ids.cpu().data.numpy()
#         # loss        = criterion(outputs, targets)

#         # for j, comment in enumerate(sampled_ids):
#         #     sampled_caption = []
#         #     for word_id in comment:
#         #         word = vocab.idx2word[word_id]
#         #         if word == '<end>':
#         #             break
#         #         elif word == '<start>' and word == '<pad>':
#         #             continue
#         #         else:
#         #             sampled_caption.append(word)

#         #     item_json = {"image_id": img_id[j], "caption":' '.join(sampled_caption) }
#         #     val_json.append(item_json)

#         # # Print log info
#         if i % args.log_step == 0:
#            print('[%d/%d] - Running model on validation set....'
#                  %(i, total_val_step))


#     path, file = os.path.split(resFile)

#     export_filename = '{}/{}_{}'.format(path, total_iterations, file)
#     print("Exporting to... {}".format(export_filename))
#     with open(export_filename, 'w') as outfile:
#         json.dump(val_json, outfile)

#     print("calculating metrics...")
#     coco = COCO(annFile)
#     cocoRes = coco.loadRes(export_filename)
#     cocoEval = COCOEvalCap(coco, cocoRes)
#     cocoEval.params['image_id'] = cocoRes.getImgIds()
#     cocoEval.evaluate()

#     for metric, score in cocoEval.eval.items():
#         log_value(metric, score, total_iterations)
#         print '%s: %.3f'%(metric, score)

#     netG.train()
#     encoder.train()
#     for parameter in encoder.parameters():
#         parameter.requires_grad=True
#     for parameter in netG.parameters():
#         parameter.requires_grad=True



            


                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=299 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab-aesthetics.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--dataset', type=str, default='aesthetics' ,
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
    parser.add_argument('--netG', type=str, default="logs/aesthetics/mle_baseline_bs16/decoder-1-10000.pkl")
    parser.add_argument('--netE', type=str, default="logs/aesthetics/mle_baseline_bs16/encoder-1-10000.pkl")

    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--clip', type=float, default=1.0,help='gradient clipping')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
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