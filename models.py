import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torchvision.models as models
import os

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
#        self.resnet = models.resnet18(pretrained=True)
#        for param in self.resnet.parameters():
#            param.requires_grad = False
#        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
#        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.inception = torch.load('net_model_epoch_10.pth').inception
        self.inception.aux_logits = False
        self.inception.transform_input = False
        for parameter in self.inception.parameters():
            parameter.requires_grad = False
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


class InceptionNet(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionNet, self).__init__()
        self.inception = models.inception_v3(pretrained=False)
        self.inception.aux_logits = False
        self.inception.transform_input = False
        self.inception.fc = nn.Linear(self.inception.fc.in_features, 10)

        # self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.inception(x)
        # x = self.log_softmax(x)

        return x

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        vocab_size = len(vocab)

        self.embed = nn.Embedding(len(self.vocab), embed_size)
        # self.embed.weight = nn.Parameter(embeddings)

        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)


        ## Tie weights
        self.embed.weight = self.linear.weight
        self.dropout = nn.Dropout(0.5)


        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.gru(packed)
        outputs = self.linear(self.dropout(hiddens[0]))
        return outputs
    
    def sample(self, features, states):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.gru(inputs, states)          # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()


    def beamSearch(self, features, states, n=5):
        inputs = features.unsqueeze(1)

        hiddens, states = self.gru(inputs, states)          # (batch_size, 1, hidden_size)
        outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
        confidences, best_choices = outputs.topk(n)

        best_list = [None] * n * n
        best_confidence = [None] * n * n
        best_states = [None] * n * n

        cached_states = [states] * n

        end_idx = self.vocab.word2idx["<end>"]
        
        best_choices = best_choices[0]
        terminated_choices = []
        terminated_confidences = []
        for i in range(50):
            for j, choice in enumerate(best_choices):
                if i != 0:
                    input_choice = choice[choice.size(0) - 1]
                else:
                    input_choice = choice
                   
                inputs = self.embed(input_choice).unsqueeze(0)
                current_confidence = confidences[:,j]

                hiddens, out_states = self.gru(inputs, cached_states[j])          # (batch_size, 1, hidden_size)
                outputs = self.linear(hiddens.squeeze(1))                         # (batch_size, vocab_size)

                # pick n best nodes for each possible choice
                inner_confidences, inner_best_choices = outputs.topk(n)
                for k, inner_choice in enumerate(inner_best_choices[0]):
                    if i != 0:
                        choice = best_choices[j]
                    item = torch.cat((choice, inner_choice))
                    # print(j * n + k)
                    best_list[j * n + k]   = item
                    best_states[j * n + k] = out_states
                    best_confidence[j * n + k] = inner_confidences[:,k] + current_confidence

            best_confidence_tensor = torch.cat(best_confidence, 0)
            _ , topk = best_confidence_tensor.topk(n - len(terminated_choices))

            topk_index = topk.data.int().cpu().numpy()

            ## Check if contains termination token ( '<end' )
            best_choices_index = [ i for i in topk_index if end_idx not in best_list[i].data.int() ]

            terminated_index   = list(set(topk_index) - set(best_choices_index))
            terminated_choices.extend([ best_list[i] for i in terminated_index ])
            terminated_confidences.extend([ best_confidence_tensor[i].data.cpu().numpy()[0] for i in terminated_index ])

            if len(best_choices_index) > 0:
                ### pick n best choices
                best_choices  = [ best_list[i] for i in best_choices_index ]
                cached_states = [ best_states[i] for i in best_choices_index ]
                confidences   = [ best_confidence_tensor[i] for i in best_choices_index ]
                confidences   = torch.cat(confidences,0).unsqueeze(0)
            else:
                break

        return terminated_choices,terminated_confidences