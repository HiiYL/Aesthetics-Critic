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
from models_began import G,D, weights_init, EncoderRNN
import pickle
import datetime
import json
from tensorboard_logger import configure, log_value
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./models/' ,
                    help='path for saving trained models')
parser.add_argument('--crop_size', type=int, default=299 ,
                    help='size for randomly cropping images')
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
parser.add_argument('--poll_step', default=1000, help="how often to poll if training has plateaued")
parser.add_argument('--patience', default=3, help="how long to wait before reducing lr")

# Model parameters
parser.add_argument('--embed_size', type=int , default=512 ,
                    help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int , default=512 ,
                    help='dimension of gru hidden states')
parser.add_argument('--num_layers', type=int , default=1 ,
                    help='number of layers in gru')
parser.add_argument('--pretrained', type=str)

parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--h', type=int, default=128, help="h value ( size of noise vector )")
parser.add_argument('--n', type=int, default=128, help="n value")
parser.add_argument('--lambda_k', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--visualize_step', default=500, help="display image frequency")
parser.add_argument('--checkpoint_step', default=50000, help="checkpoint frequency")

parser.add_argument('--image_size', default=128, help="size of image to generate")


parser.add_argument('--netG', type=str)
parser.add_argument('--netD', type=str)

parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--clip', type=float, default=1.0,help='gradient clipping')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
args = parser.parse_args()
print(args)

transform = transforms.Compose([
    transforms.Scale((args.image_size,args.image_size)),
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
                         transform, 1,
                         shuffle=False, num_workers=args.num_workers)

print('===> Building model')
if args.netG:
    netG = torch.load(args.netG)
    print('==> Loaded model.')
    for parameter in netG.parameters():
        parameter.requires_grad = True
else:
    netG = G(h=args.h, n=args.n, output_dim=(3,args.image_size,args.image_size))
    netG.apply(weights_init)

if args.netD:
    netD = torch.load(args.netD)
    print('==> Loaded model.')
    for parameter in netD.parameters():
        parameter.requires_grad = True
else:
    netD = D(h=args.h, n=args.n, input_dim=(3,args.image_size,args.image_size))
    netD.apply(weights_init)


netR = EncoderRNN(args.embed_size, args.hidden_size, vocab)
z_D = torch.FloatTensor(args.batch_size, args.h)
z_G = torch.FloatTensor(args.batch_size, args.h)

if True:
    netG = netG.cuda()
    netD = netD.cuda()
    netR = netR.cuda()
    z_D, z_G = z_D.cuda(), z_G.cuda()


z_D = Variable(z_D)
z_G = Variable(z_G)
optimizerG = torch.optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
optimizerD = torch.optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
optimizerR = torch.optim.Adam(netR.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

class AverageTracker():
    def __init__(self):
        self.reset()

    def add(self,x):
        self.sum += x
        self.count += 1

    def get_average(self):
        return self.sum / self.count

    def reset(self):
        self.count = 0
        self.sum   = 0

def train():
    total_iterations=0
    k_t=0
    fixed_sample = None
    fixed_x = None
    fixed_caption = None
    best_measure = 1e7
    patience = args.patience
    measure_tracker = AverageTracker()

    for i in range(999):
        for iteration, (images, captions, lengths, img_id, wrong_captions, wrong_lengths) in enumerate(train_data_loader):
            # Set mini-batch dataset
            X = Variable(images)
            captions = Variable(captions)
            wrong_captions = Variable(wrong_captions)
            if torch.cuda.is_available():
                X = X.cuda()
                captions = captions.cuda()
                wrong_captions = wrong_captions.cuda()

            encoded_caption = netR(captions, lengths)
            encoded_wrong_caption = netR(wrong_captions, wrong_lengths)
            
            netD.zero_grad()
            netG.zero_grad()
            netR.zero_grad()

            z_D.data.normal_(0,1)
            z_D_caption = torch.cat((z_D, encoded_caption), 1)
            z_D_wrong_caption = torch.cat((z_D, encoded_wrong_caption), 1)

            G_zD = netG(z_D_caption.detach())
            AE_x = netD(X, encoded_caption)
            AE_x_wrong = netD(X, encoded_wrong_caption)
            AE_G_zD = netD(G_zD.detach(), encoded_caption)

            d_loss_real = torch.mean(torch.abs(AE_x - X))
            d_loss_wrong = torch.mean(torch.abs(AE_x_wrong - X))
            d_loss_fake = torch.mean(torch.abs(AE_G_zD - G_zD.detach()))

            D_loss = d_loss_real - k_t * 0.5 * ( d_loss_fake + d_loss_wrong )
            D_loss.backward()
            optimizerD.step()

            netG.zero_grad()

            encoded_caption = netR(captions, lengths)
            z_G.data.normal_(0,1)
            z_G_caption = torch.cat((z_G, encoded_caption), 1)

            G_zG = netG(z_G_caption)
            AE_G_zG = netD(G_zG, encoded_caption)
            G_loss = torch.mean(torch.abs(G_zG - AE_G_zG.detach()))
            G_loss.backward()

            optimizerG.step()
            optimizerR.step()

            if fixed_sample is None:
                fixed_caption = Variable(encoded_caption.data.clone(), volatile=True)
                fixed_sample = Variable(z_G_caption.data.clone(), volatile=True)
                fixed_x = Variable(X.data.clone(), volatile=True)
                vutils.save_image(X.data, '{}/x_fixed.jpg'.format(save_path), normalize=True,range=(-1,1))

            balance = ( args.gamma * d_loss_real - G_loss ).data[0]
            measure = d_loss_real.data[0] + abs(balance)

            k_t += args.lambda_k * balance
            k_t = max(min(1, k_t), 0)

            if total_iterations % args.log_step == 0:
                print("===> Epoch[{}]({}/{}): D_Loss: {:.4f} | G_Loss: {:.4f} | Measure: {:.4f} | k_t: {:.4f}".format(
                    epoch, iteration, len(train_data_loader), D_loss.data[0], G_loss.data[0], measure,k_t))

            total_iterations += 1
            if total_iterations % args.tb_log_step == 0:
                log_value('D_Loss', D_loss.data[0], total_iterations)
                log_value('G_Loss', G_loss.data[0], total_iterations)
                log_value('Measure', measure, total_iterations)
                log_value('k', k_t, total_iterations)

            if (total_iterations % args.visualize_step == 0) or total_iterations == 1:
                ae_x = netD(fixed_x, fixed_caption)
                g = netG(fixed_sample)
                ae_g = netD(g, fixed_caption)
                
                vutils.save_image(ae_g.data, '{}/{}_D_fake.jpg'.format(save_path, total_iterations), normalize=True,range=(-1,1))
                vutils.save_image(ae_x.data, '{}/{}_D_real.jpg'.format(save_path, total_iterations), normalize=True,range=(-1,1))
                vutils.save_image(g.data, '{}/{}_G.jpg'.format(save_path, total_iterations), normalize=True,range=(-1,1))

            if total_iterations % args.checkpoint_step == 0:
                checkpoint(total_iterations, save_path)

            measure_tracker.add(measure)
            if total_iterations % args.poll_step == 0:
                measure_avg = measure_tracker.get_average()
                if measure_avg < best_measure:
                    best_measure = measure
                else:
                    patience -= 1
                    print("[!] Measure not decreasing | Best : {} | Current: {} | Patience: {}"
                        .format(best_measure, measure_avg, patience ))
                    if patience <= 0:
                        patience = args.patience
                        times_reduced_lr += 1
                        lr =  args.lr * (0.5 ** times_reduced_lr)
                        print("[!] Reducing lr to {} at iteration {}".format(lr, total_iterations))
                        for param_group in argsimizerG.param_groups:
                            param_group['lr'] = lr
                        for param_group in argsimizerD.param_groups:
                            param_group['lr'] = lr
                measure_tracker.reset()


def checkpoint(epoch, save_path):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", args.dataset)):
        os.mkdir(os.path.join("checkpoint", args.dataset))

    now = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
    netG_model_out_path = "{}/netG_model_iter_{}.pth".format(save_path,epoch)
    netD_model_out_path = "{}/netD_model_iter_{}.pth".format(save_path,epoch)
    netR_model_out_path = "{}/netR_model_iter_{}.pth".format(save_path,epoch)
    torch.save(netG, netG_model_out_path)
    torch.save(netD, netD_model_out_path)
    torch.save(netR, netR_model_out_path)
    print("Checkpoint saved to {}".format(save_path))





if not os.path.exists("logs"):
    os.mkdir("logs")

if not os.path.exists(os.path.join("logs", args.dataset)):
    os.mkdir(os.path.join("logs", args.dataset))

now = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
save_path = os.path.join(os.path.join("logs", args.dataset), now)

if not os.path.exists(save_path):
    os.mkdir(save_path)

configure(save_path)
for epoch in range(1, 999 + 1):
    netG.train()
    netD.train()
    train()