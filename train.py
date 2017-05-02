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
from data_loader import get_loader 
from build_vocab import Vocabulary
from models import EncoderCNN, DecoderRNN,InceptionNet
import pickle
import datetime

from tensorboard_logger import configure, log_value

def train(save_path, args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    

    training_location = "data/{}/train".format(args.dataset)
    # Build data loader
    data_loader = get_loader(training_location, args.comments_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models
    encoder = EncoderCNN(args.embed_size, torch.load('data/net_model_epoch_10.pth').inception)
    encoder.set_finetune(finetune=True)

    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                             vocab, args.num_layers)

    if args.pretrained:
        encoder.load_state_dict(torch.load('models/encoder{}'.format(args.pretrained)))
        decoder.load_state_dict(torch.load('models/decoder{}'.format(args.pretrained)))

    state = (Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)),
     Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size)))

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        state = [s.cuda() for s in state]

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    #fc_params = list(map(id, encoder.inception.fc.parameters()))
    #base_params = filter(lambda p: id(p) not in ignored_params,
    #                 encoder..parameters())
    params = [
                {'params': decoder.parameters()},
                {'params': encoder.parameters(), 'lr': 0.1 * args.learning_rate}
            ]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the Models
    total_step = len(data_loader)

    total_iterations = 0

    features_fixed = None
    captions_fixed = None
    lengths_fixed = None

    for epoch in range(args.num_epochs):
        if epoch % 8 == 0:
            lr = args.learning_rate * (0.5 ** (epoch // 8))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, (images, captions, lengths) in enumerate(data_loader):    
            # Set mini-batch dataset
            images = Variable(images)
            captions = Variable(captions)
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            targets, batch_sizes = pack_padded_sequence(captions, lengths, batch_first=True)

            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs  = decoder(features, captions, lengths, state, teacher_forced=True)

            loss = 0
            for i in range(outputs.size(0)):
                loss += ( 0.9 ** ( i + 1 ) ) * criterion(outputs[i], targets[i])
            # loss     = criterion(outputs, targets)

            loss.backward()
            #torch.nn.utils.clip_grad_norm(decoder.parameters(), args.clip)
            optimizer.step()

            # Print log info
            if total_iterations % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 

            if total_iterations % 100 == 0:
                print()
                decoder.eval()
                for parameter in decoder.parameters():
                    parameter.requires_grad=False
                outputs  = decoder(features, captions, lengths, state, teacher_forced=False, use_random=False)
                for parameter in decoder.parameters():
                    parameter.requires_grad=True
                decoder.train()
                sampled_ids = torch.max(outputs,1)[1].squeeze()
                sampled_ids = pad_packed_sequence([sampled_ids, batch_sizes], batch_first=True)[0]
                sampled_ids = sampled_ids.cpu().data.numpy()

                for i, comment in enumerate(sampled_ids):
                    if i > 1:
                        break
                    sampled_caption = []
                    for word_id in comment:
                        word = vocab.idx2word[word_id]
                        sampled_caption.append(word)
                        if word == '<end>':
                            break
                    sentence = "[P]" + ' '.join(sampled_caption)
                    print(sentence)

                    sampled_caption = []
                    sample = captions.cpu().data.numpy()
                    for word_id in sample[i]:
                        word = vocab.idx2word[word_id]
                        sampled_caption.append(word)
                        if word == '<end>':
                            break
                    sentence = "[S]" + ' '.join(sampled_caption)
                    print(sentence)
                    print()

            # Save the model
            if (total_iterations+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), 
                           os.path.join(save_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(save_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))


            if total_iterations % args.tb_log_step == 0:
                log_value('Loss', loss.data[0], total_iterations)
                log_value('Perplexity', np.exp(loss.data[0]), total_iterations)

            total_iterations += 1


                
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
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('--pretrained', type=str)#, default='-2-20000.pkl')
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
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