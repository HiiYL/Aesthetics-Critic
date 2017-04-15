import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models
import os

pretrained_path = "squeezenet1_1-f364aa15.pth"
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
#        self.resnet = models.resnet18(pretrained=True)
#        for param in self.resnet.parameters():
#            param.requires_grad = False
#        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
#        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        self.inception = models.inception_v3(pretrained=True)
        self.inception.aux_logits = False
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.inception.fc.weight.data.normal_(0.0, 0.02)
        self.inception.fc.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.inception(images)
        #features = self.bn(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        vocab_size = len(vocab)
        # self.embed = nn.Embedding(vocab_size, embed_size)

        matrix_filepath='data/embedding_matrix.npy'
        if not os.path.isfile(matrix_filepath):
            print("generating embedding...")
            embeddings_index = self.generateIndexMappingToEmbedding()
            embedding_matrix = np.zeros((len(self.vocab) + 1, 300))
            for word, i in vocab.word2idx.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            embedding_matrix.tofile(matrix_filepath)
        else:
            embedding_matrix = np.fromfile(matrix_filepath)
            embedding_matrix = embedding_matrix.reshape((len(self.vocab) + 1, 300))

        embeddings = torch.from_numpy(embedding_matrix).float()

        self.embed = nn.Embedding(embeddings.size(0), embeddings.size(1))
        self.embed.weight = nn.Parameter(embeddings)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        # self.embed.weight.data.uniform_(-0.1, 0.1)

        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(50):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()

    def generateIndexMappingToEmbedding():
        filename = 'glove.6B.300d.txt'
        print("Using pretrained GloVe from : {}".format(filename))
        embeddings_index = {}
        with open(filename,'r') as f:
          for line in f:
              values = line.split()
              word = values[0]
              coefs = np.asarray(values[1:], dtype='float32')
              embeddings_index[word] = coefs
        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ELU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ELU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

def squeezenet1_1(pretrained=False, **kwargs):
    model = SqueezeNet()
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
            
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2)
        # self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)


        self.fire2 = Fire(96, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        self.fire4 = Fire(128, 32, 128, 128)
        self.fire5 = Fire(256, 32, 128, 128)
        #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)


        self.conv2 = nn.Conv2d(512, 1024, kernel_size=1)

        self.dropout = nn.Dropout(p=0.5)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.linear = nn.Linear(1024, num_classes)

        ##
        # https://discuss.pytorch.org/t/kullback-leibler-divergence-loss-function-giving-negative-values/763
        ##
        self.softmax = nn.LogSoftmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = 2.0
                if m is self.conv2:
                    m.weight.data.normal_(0, 0.01)
                else:
                    fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    u = math.sqrt(3.0 * gain / fan_in)
                    m.weight.data.uniform_(-u, u)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.maxpool(self.elu(self.conv1(x)))
        x = self.fire2(x)
        x_s = self.fire3(x)

        x = self.maxpool(self.fire4(torch.add(x, x_s)))
        x_s = self.fire5(x)
        x = self.fire6(torch.add(x,x_s))
        x_s = self.fire7(x)

        x = self.maxpool(self.fire8(torch.add(x,x_s)))
        x_s = self.dropout(self.fire9(x))

        x = self.elu(self.conv2(torch.add(x,x_s)))

        x = self.gap(x).view(1,-1)
        # print(x)
        return x
