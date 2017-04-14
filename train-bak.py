from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from models import SqueezeNet

from tensorboard_logger import configure, log_value


import numpy as np

import datetime


# python train.py --dataset aesthetics-unscaled --cuda --batchSize 1 --testBatchSize 1
configure("runs/aesthetics-{}".format(datetime.datetime.now()))

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batchSize', type=int, default=16, help='raining batch size')
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--net', default='', help="path to net (to continue training)")
# Model parameters
parser.add_argument('--embed_size', type=int , default=256 ,help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int , default=512 ,help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int , default=1 ,help='number of layers in lstm')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"



train_set = get_training_set(root_path + opt.dataset)

test_set = get_test_set(root_path + opt.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if opt.net:
    net = torch.load(opt.net)
    print('==> Loaded model.')
    for parameter in net:
        parameter.requires_grad = True
else:
    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    # net.classifier._modules['1'] = nn.Conv2d(512,10, kernel_size=(1, 1), stride=(1, 1))
    # net.num_classes = 10

    # net = models.vgg16(pretrained=True)
    # net.classifier._modules['6'] = None
    # net.classifier._modules['6'] = nn.Linear(4096, 1)


criterion_kld = nn.KLDivLoss()

real_A = torch.FloatTensor(opt.batchSize, 3, 224, 224)
captions = torch.FloatTensor(opt.batchSize, 10)
# semantic_label = torch.LongTensor(opt.batchSize, 1)
# semantic_label_onehot = torch.FloatTensor(opt.batchSize, 20)

if opt.cuda:
    # vgg = vgg.cuda()
    # sn = sn.cuda()
    net = net.cuda()
    criterion_kld = criterion_kld.cuda()
    real_A = real_A.cuda()
    captions = captions.cuda()
    # semantic_label = semantic_label.cuda()
    # semantic_label_onehot = semantic_label_onehot.cuda()

real_A = Variable(real_A)
captions_v = Variable(captions)
# semantic_label = Variable(semantic_label)
# semantic_label_onehot = Variable(semantic_label_onehot)

# setup optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0001)
# optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

total_iterations=0
def train(epoch):
    global total_iterations
    for i, (images, captions, lengths) in enumerate(training_data_loader):

        # real_a_cpu, aesthetics_label_batch = batch[0], batch[1]

        real_A.data.resize_(images.size()).copy_(images)
        captions_v.data.resize_(captions.size()).copy_(captions)

        net.zero_grad()
        aesthetics_pred = net(real_A)

        loss = criterion_kld(aesthetics_pred, aesthetics_label)
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
        optimizer.step()

        if iteration % 1000 == 1:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss.data[0]))


        # if iteration % 500 == 1:
        #     total_iterations += iteration
            # log_value('Loss', loss.data[0], total_iterations)
    # log_value('training_loss', loss.data[0], epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_accuracy(input, label):
    x = input.data.cpu().numpy()
    y = label.data.cpu().numpy()
    weights = np.array([1,2,3,4,5,6,7,8,9,10])
    if len(x.shape) == 1:
        x = x[np.newaxis, :]

    score_pred = (x * weights).sum(axis=1)
    score_test = (y * weights).sum(axis=1)

    Y_pred_binary = np.array([ 1 if row >= 5 else 0 for row in score_pred])
    Y_test_binary = np.array([ 1 if row >= 5 else 0 for row in score_test])

    # print()
    # print(Y_pred_binary)
    # print(Y_test_binary)
    # print()

    accuracy = np.sum(Y_pred_binary == Y_test_binary) / len(Y_test_binary)

    return accuracy



def test(epoch):
    print("testing ... ")
    for p in net.parameters():
        p.requires_grad = False

    meter = AverageMeter()
    testing_len = len(testing_data_loader)

    for iteration, batch in enumerate(testing_data_loader, 1):

        real_a_cpu, aesthetics_label_batch = batch[0], batch[1]

        ## GT Image
        real_A.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)

        ## Label
        aesthetics_label.data.resize_(aesthetics_label_batch.size()).copy_(aesthetics_label_batch)

        aesthetics_pred = net(real_A)

        pred_prob = torch.exp(aesthetics_pred)
        # print(aesthetics_label)
        # print(pred_prob)

        accuracy = eval_accuracy(pred_prob, aesthetics_label)
        meter.update(accuracy)
        if iteration % 1000 == 0:
            print("[ {} / {} ] acc: {} | avg: {}".format(iteration, testing_len, meter.val, meter.avg))

    print("Final Accuracy = {} %".format(meter.avg))
    log_value('classification_accuracy', meter.avg, epoch)

    for p in net.parameters():
        p.requires_grad = True

def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))
    net_model_out_path = "checkpoint/{}/net_model_epoch_{}.pth".format(opt.dataset, epoch)
    torch.save(net, net_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test(epoch)


    if epoch % 5 == 0:
        checkpoint(epoch)
