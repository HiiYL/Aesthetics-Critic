import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torch.autograd import Variable 
from torchvision import transforms 
from build_vocab import Vocabulary
from models import EncoderCNN, DecoderRNN
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

    # Build Models
    encoder = EncoderCNN(args.embed_size)
    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         vocab, args.num_layers)
    decoder.eval()
    

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=lambda storage, loc: storage))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=lambda storage, loc: storage))

    # Prepare Image       
    image = Image.open(args.image)
    image_tensor = Variable(transform(image).unsqueeze(0), volatile=True)

    encoder.inception.transform_input = False

    #print(image_tensor)
    
    # Set initial states
    state = (Variable(torch.zeros(args.num_layers, 1, args.hidden_size)),
            Variable(torch.zeros(args.num_layers, 1, args.hidden_size)))
    
    # If use gpu
    if torch.cuda.is_available():
       encoder.cuda()
       decoder.cuda()
       state = [s.cuda() for s in state]
       image_tensor = image_tensor.cuda()
    
    # Generate caption from image
    feature = encoder(image_tensor)

    sampled_ids_list, terminated_confidences = decoder.beamSearch(feature, None, n = args.n)
    sampled_ids_list = np.array([ sampled_ids.cpu().data.numpy() for sampled_ids in sampled_ids_list ])

    # sampled_ids_list = [x for (y,x) in sorted(zip(terminated_confidences,sampled_ids_list),reverse=True)]
    
    # Decode word_ids to words
    sorted_idx = np.argsort(terminated_confidences)

    sampled_ids_sorted = sorted(zip(terminated_confidences,sampled_ids_list), reverse=True)
    #sampled_ids_sorted = [x for (y,x) in sorted(zip(terminated_confidences,sampled_ids_list))]
    
    for _, sampled_with_confidence in enumerate(sampled_ids_sorted):
        confidence, sample  = sampled_with_confidence
        sampled_caption = []
        for word_id in sample:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        
        # Print out image and generated caption.
        print ("{} - {:.4f} ".format(sentence, confidence))
    plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-2-10000.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-2-10000.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--crop_size', type=int, default=299,
                        help='size for center cropping images')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2 ,
                        help='number of layers in lstm')
    parser.add_argument('--n', type=int , default=5 ,
                        help='n of beam search')
    args = parser.parse_args()
    main(args)