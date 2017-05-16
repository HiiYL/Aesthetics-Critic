import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torch.autograd import Variable 
from torchvision import transforms 
from build_vocab import Vocabulary
from models_spatial import EncoderCNN, G_Spatial
from torchvision import models
from PIL import Image


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Scale((args.crop_size, args.crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build the models
    encoder = EncoderCNN(args.embed_size,models.inception_v3(pretrained=True), requires_grad=False)
    netG = G_Spatial(args.embed_size, args.hidden_size, vocab, args.num_layers)

    if args.encoder:
        print("[!]loading pretrained decoder....")
        encoder.load_state_dict(torch.load(args.encoder))
        print("Done!")


    if args.netG:
        print("[!]loading pretrained netG....")
        netG.load_state_dict(torch.load(args.netG))
        print("Done!")



    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    netG.eval()

    # Prepare Image       
    image = Image.open(args.image)
    image = Variable(transform(image).unsqueeze(0), volatile=True)

    #print(image_tensor)
    
    # Set initial states
    state = (Variable(torch.zeros(args.num_layers, 1, args.hidden_size)),
            Variable(torch.zeros(args.num_layers, 1, args.hidden_size)))
    
    # If use gpu
    if torch.cuda.is_available():
       encoder.cuda()
       netG.cuda()
       state = [s.cuda() for s in state]
       image = image.cuda()

    features = encoder(image)
    features_g, features_l = netG.encode_fc(features)

    sampled_ids_list, terminated_confidences = netG.beamSearch((features_g, features_l), state, n=10, diverse_gamma=0.0)
    sampled_ids_list = np.array([ sampled_ids.cpu().data.numpy() for sampled_ids in sampled_ids_list ])

    max_index = np.argmax(terminated_confidences)
    
    #print(max_index)
    sampled_ids = sampled_ids_list[max_index]
    def word_idx_to_sentence(sample):
        sampled_caption = []
        for word_id in sample:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        return ' '.join(sampled_caption)

    for sampled_ids in sampled_ids_list:
        print(word_idx_to_sentence(sampled_ids))
    #print(word_idx_to_sentence(sampled_ids))

    # sampled_ids_list, terminated_confidences = decoder.beamSearch(feature, None, n = args.n)

    # sampled_ids_list = np.array([ sampled_ids.cpu().data.numpy() for sampled_ids in sampled_ids_list ])

    # # sampled_ids_list = [x for (y,x) in sorted(zip(terminated_confidences,sampled_ids_list),reverse=True)]
    
    # # Decode word_ids to words
    # sorted_idx = np.argsort(terminated_confidences)

    # sampled_ids_sorted = sorted(zip(terminated_confidences,sampled_ids_list))
    # #sampled_ids_sorted = [x for (y,x) in sorted(zip(terminated_confidences,sampled_ids_list))]
    
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')
    parser.add_argument('--encoder', type=str, default='logs/aesthetics/16052017152815/encoder-1-8351.pkl',
                        help='path for trained encoder')
    parser.add_argument('--netG', type=str, default='logs/aesthetics/16052017152815/netG-1-8351.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab-aesthetics.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--crop_size', type=int, default=299,
                        help='size for center cropping images')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of gru hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in gru')
    parser.add_argument('--n', type=int , default=5 ,
                        help='n of beam search')
    args = parser.parse_args()
    main(args)