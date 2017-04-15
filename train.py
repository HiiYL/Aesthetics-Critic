import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from data_loader import get_loader 
from build_vocab import Vocabulary
from models import EncoderCNN, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

import pickle

def main(args):
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
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.comments_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models

    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                             vocab, args.num_layers)

    if args.pretrained:
        encoder.load_state_dict(torch.load('models/encoder{}'.format(args.pretrained)))
        decoder.load_state_dict(torch.load('models/decoder{}'.format(args.pretrained)))


    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    #fc_params = list(map(id, encoder.inception.fc.parameters()))
    #base_params = filter(lambda p: id(p) not in ignored_params,
    #                 encoder..parameters())
    params = [
                {'params': decoder.parameters()},
                {'params': encoder.inception.fc.parameters()}
            ]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the Models
    total_step = len(data_loader)

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
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions, lengths)

            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm(decoder.parameters(), args.clip)
            optimizer.step()

            #print(targets)
            #print(outputs)
            #exit()


            #print()
            #print()


            # Print log info
            if i % args.log_step == 0:

                #sampled_caption = []
                #for idx, word_id in enumerate(outputs):
                #    if idx % 8 == 0:
                #        word = vocab.idx2word[np.argmax(word_id.data.cpu().numpy())]
                #        sampled_caption.append(word)
                #        if word == '<end>':
                #            break
                #sentence = ' '.join(sampled_caption)
                #print(sentence)
                #print()

                #sampled_caption = []
                #for idx, word_id in enumerate(targets.data.cpu().numpy()):
                #    if idx % 8 == 0:
                #        word = vocab.idx2word[word_id]
                #        sampled_caption.append(word)
                #        if word == '<end>':
                #            break
                #sentence = ' '.join(sampled_caption)
                #print(sentence)
                #print()


                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
            
            # if (i % (args.log_step * 10))  == 0:
            #     print()
            #     print()
            #     sampled_ids = decoder.sample(features)
            #     sampled_ids = sampled_ids.cpu().data.numpy()
                
            #     # Decode word_ids to words
            #     sampled_caption = []
            #     for word_id in sampled_ids[0]:
            #         word = vocab.idx2word[word_id]
            #         sampled_caption.append(word)
            #         if word == '<end>':
            #             break
            #     sentence = ' '.join(sampled_caption)
            #     print(sentence)
            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=299 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/aesthetics/train' ,
                        help='directory for resized images')
    parser.add_argument('--comments_path', type=str,
                        default='labels.h5',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=10000,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2 ,
                        help='number of layers in lstm')
    parser.add_argument('--pretrained', type=str)#, default='-2-20000.pkl')
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clip', type=float, default=0.25,help='gradient clipping')
    args = parser.parse_args()
    print(args)
    main(args)