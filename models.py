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
import torch.nn.functional as F
import random
class EncoderCNN(nn.Module):
    def __init__(self, embed_size,inception):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
#        self.resnet = models.resnet18(pretrained=True)
#        for param in self.resnet.parameters():
#            param.requires_grad = False
#        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
#        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.aux_logits = False
        self.transform_input = False
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        # if aux_logits:
        #     self.AuxLogits = inception.AuxLogits
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.fc = nn.Linear(inception.fc.in_features, embed_size)#inception.fc

        self.init_weights()

    def set_finetune(self,finetune=False):
        if finetune:
            for parameters in self.parameters():
                parameters.requires_grad = True
        else:
            for parameters in self.parameters():
                parameters.requires_grad = False
            for parameters in self.fc.parameters():
                parameters.requires_grad = True

        
    def init_weights(self):
        """Initialize the weights."""
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)
        
    def forward(self, x):
        """Extract the image feature vectors."""
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])

        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)

        return x

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.embed = nn.Embedding(self.vocab_size, embed_size)
        # self.embed.weight = nn.Parameter(embeddings)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,dropout=0.5, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

        ## Tie weights
        #self.embed.weight = self.linear.weight
        self.dropout = nn.Dropout(0.5)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths, state):
        """Decode image feature vectors and generates captions."""
        if random.random() > 0.5:
            return self.forward_forced(features, captions, lengths, state)
        else:
            return self.forward_free(features, captions, lengths, state)

    def forward_forced(self,features, captions, lengths, state):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        hiddens, _ = self.lstm(packed)
        outputs = self.linear(self.dropout(hiddens[0]))

        return outputs

    def forward_free(self, features, captions, lengths, states):
        #hiddens = torch.FloatTensor(2,lengths[0],512)
        # output_list = []
        output_tensor = Variable(torch.cuda.FloatTensor(1, lengths[0], 43892))
        inputs = features.unsqueeze(1)
        for i in range(lengths[0]):
            #inputs = embeddings[:,i,:].unsqueeze(1)
            hiddens, states = self.lstm(inputs, states)
            output = self.linear(self.dropout(hiddens.squeeze(1)))
            inputs = self.embed(output.max(1)[1])
            output_tensor[:,i,:] = output.unsqueeze(1)
            
        output_packed = pack_padded_sequence(output_tensor, lengths, batch_first=True)

        return output_packed[0] 

    # def forward_free(self, features, captions, lengths, states):
    #     #hiddens = torch.FloatTensor(2,lengths[0],512)
    #     hiddens_list = []

    #     inputs = features.unsqueeze(1)
    #     for i in range(lengths[0]):
    #         #inputs = embeddings[:,i,:].unsqueeze(1)
    #         hiddens, states = self.lstm(inputs, states)
    #         hiddens_list.append(hiddens)
    #         inputs = hiddens
    #         #hiddens[:,i,:] = hiddens.squeeze()
            
    #         #predicted = outputs.max(1)[1]
    #         #embeddings = self.embed(predicted)

    #     hiddens_list = torch.cat(hiddens_list, 1)
    #     hidden_packed = pack_padded_sequence(hiddens_list, lengths, batch_first=True)

    #     #packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
    #     #hiddens, _ = self.lstm(packed)
    #     #print(hiddens)

    #     outputs = self.linear(self.dropout(hidden_packed[0]))

    #     return outputs 
    
    def sample(self, features, states):
        """Samples captions for given image features (Greedy search)."""
        # sampled_ids = []
        hiddens_list = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size)
            # outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            hiddens_list.append(hiddens)
            # predicted = outputs.max(1)[1]
            # sampled_ids.append(predicted)
            inputs = hiddens
            #inputs = self.embed(predicted)
        # sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        hiddens_list = torch.cat(hiddens_list, 1)

        outputs = self.linear(self.dropout(hiddens_list[0]))
        # print(torch.max(outputs,1)[1])
        # exit()
        return torch.max(outputs,1)[1].squeeze()   #sampled_ids.squeeze()


    def beamSearch(self, features, states, n=5):
        inputs = features.unsqueeze(1)

        hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size)
        outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
        confidences, best_choices = outputs.topk(n)



        cached_states = [states] * n

        end_idx = self.vocab.word2idx["<end>"]
        
        best_choices = best_choices[0]
        terminated_choices = []
        terminated_confidences = []

        # one loop for each word
        for word_index in range(50):
            #print(best_choices)

            best_list = [None] * n * (n - len(terminated_choices))
            best_confidence = [None] * n * (n - len(terminated_choices))
            best_states = [None] * n * (n - len(terminated_choices))

            # for each choice
            for choice_index, choice in enumerate(best_choices):

                input_choice = choice[word_index] if word_index != 0 else choice
                   
                inputs = self.embed(input_choice).unsqueeze(0)
                current_confidence = confidences[:,choice_index]

                hiddens, out_states = self.lstm(inputs, cached_states[choice_index])        # (batch_size, 1, hidden_size)
                outputs = self.linear(hiddens.squeeze(1))                                   # (batch_size, vocab_size)

                # pick n best nodes for each possible choice
                inner_confidences, inner_best_choices = outputs.topk(n)

                inner_confidences = inner_confidences[0]
                for inner_choice_idx, inner_choice in enumerate(inner_best_choices[0]):
                    if word_index != 0:
                        choice = best_choices[choice_index]
                    item = torch.cat((choice, inner_choice))

                    position = choice_index * n + inner_choice_idx

                    best_list[position]   = item
                    best_states[position] = out_states
                    best_confidence[position] = inner_confidences[inner_choice_idx] + current_confidence


            best_confidence_tensor = torch.cat(best_confidence, 0)
            _ , topk = best_confidence_tensor.topk(n - len(terminated_choices))

            topk_index = topk.data.int().cpu().numpy()

            ## Filter nodes that contains termination token '<end>'
            best_choices_index = [ index for index in topk_index if end_idx not in best_list[index].data.int() ]

            terminated_index   = list(set(topk_index) - set(best_choices_index))
            terminated_choices.extend([ best_list[index] for index in terminated_index ])
            terminated_confidences.extend([ best_confidence_tensor[index].data.cpu().numpy()[0] for index in terminated_index ])


            ## If there is still nodes to evaluate
            if len(best_choices_index) > 0:
                best_choices  = [ best_list[index] for index in best_choices_index ]
                cached_states = [ best_states[index] for index in best_choices_index ]
                confidences   = [ best_confidence_tensor[index] for index in best_choices_index ]
                confidences   = torch.cat(confidences,0).unsqueeze(0)
            else:
                break

        #print(best_choices)
        if len(best_choices_index) > 0:
            terminated_choices.extend([ best_list[index] for index in best_choices_index ])
            terminated_confidences.extend([ best_confidence_tensor[index].data.cpu().numpy()[0] for index in best_choices_index ])
        return terminated_choices,terminated_confidences


class ValueNetwork(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(ValueNetwork, self).__init__()
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
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.gru(packed)
        outputs = self.linear(self.dropout(hiddens[0]))

        return outputs


class InceptionNet(nn.Module):

    def __init__(self, num_classes=10):
        super(InceptionNet, self).__init__()
        self.inception = models.inception_v3(pretrained=False)
        self.inception.aux_logits = False
        self.inception.transform_input = False
        self.inception.fc = nn.Linear(self.inception.fc.in_features, 10)

        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.inception(x)
        x = self.log_softmax(x)

        return x