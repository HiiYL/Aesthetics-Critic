import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,PackedSequence
import torchvision.models as models
import os
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size,inception, requires_grad=False):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
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

        for parameters in self.parameters():
            parameters.requires_grad = requires_grad
        
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
        #x = F.avg_pool2d(x, kernel_size=x.size()[2:])

        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        # x = x.view(x.size(0), -1)
        # 2048

        return x

class G(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(G, self).__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.embed = nn.Embedding(self.vocab_size, embed_size)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

        #self.adaptive_pool = nn.AdaptiveMaxPool2d((8,8))
        self.conv = nn.Conv2d(2048, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        self.hidden_size = hidden_size

        self.gru_cell = nn.GRUCell(embed_size, hidden_size)
        
        self.linear_fc = nn.Linear(hidden_size * 2, hidden_size)

        self.attn = nn.Linear( hidden_size * 2, hidden_size)
        self.attn_combine = nn.Linear( hidden_size * 2, hidden_size)

        self.log_softmax = nn.LogSoftmax()
        self.relu = nn.ReLU()

        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)

        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

        self.attn.weight.data.normal_(0.0, 0.02)
        self.attn.bias.data.fill_(0)

        self.attn_combine.weight.data.normal_(0.0, 0.02)
        self.attn_combine.bias.data.fill_(0)


    def gru_attention(self, inputs, hx, features):
        attn_weights = F.softmax(self.attn(torch.cat((inputs, hx), 1)))
        attn_applied = features * attn_weights
        inputs = self.relu(self.attn_combine(torch.cat((inputs, attn_applied ), 1)))

        hx = self.gru_cell(inputs, hx)

        return hx

         
    def forward(self, features, captions, lengths, states, teacher_forced=True):
        """Decode image feature vectors and generates captions."""
        #features = self.adaptive_pool(features)
        features = self.conv(features)
        features = features.view(features.size(0), -1)
        features = self.bn(self.fc(features))
        # return self.forward_forced(features, captions,lengths)
        if teacher_forced:
            return self._forward_forced_cell(features, captions,lengths, states)
        else:
            return self._forward_free_cell(features,lengths, states)

    def _forward_forced_cell(self, features, captions, lengths, states):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens_tensor = Variable(torch.cuda.FloatTensor(len(lengths),lengths[0],self.hidden_size))
        hx = states[0].squeeze()
        for i in range(lengths[0]):
            inputs = embeddings[:,i,:]
            if hx.data.size(0) > inputs.data.size(0):
                hx = hx[:inputs.size(0)]
                
            hx = self.gru_attention(inputs, hx, features)
            hiddens_tensor[ :, i, :] = hx

        hiddens_tensor_packed, _ = pack_padded_sequence(hiddens_tensor, lengths, batch_first=True)
        outputs = self.linear(hiddens_tensor_packed)
        return self.log_softmax(outputs), hiddens_tensor, embeddings

    def _forward_free_cell(self, features, lengths, states):
        output_tensor = Variable(torch.cuda.FloatTensor(len(lengths),lengths[0],self.vocab_size))
        hiddens_tensor = Variable(torch.cuda.FloatTensor(len(lengths),lengths[0],self.hidden_size))

        inputs = features
        hx = states[0].squeeze()
        for i in range(lengths[0]):
            if hx.data.size(0) > inputs.data.size(0):
                hx = hx[:inputs.size(0)]

            hx = self.gru_attention(inputs, hx, features)
            outputs = self.log_softmax(self.linear(hx))
            predicted = outputs.max(1)[1]
            inputs = self.embed(predicted).view(predicted.size(0), -1)

            hiddens_tensor[:,i,:] = hx
            output_tensor [:,i,:] = outputs

        output_tensor, _ = pack_padded_sequence(output_tensor, lengths, batch_first=True)
        #hiddens_tensor, _ = pack_padded_sequence(hiddens_tensor, lengths, batch_first=True)
        return output_tensor, hiddens_tensor

    def sample(self, features, states):
        features = self.conv(features)
        features = features.view(features.size(0), -1)
        features = self.bn(self.fc(features))

        inputs = features
        sampled_ids = []
        hx = states[0].squeeze()
        for i in range(50):
            if hx.data.size(0) > inputs.data.size(0):
                hx = hx[:inputs.size(0)]

            hx = self.gru_attention(inputs, hx, features)
            outputs = self.log_softmax(self.linear(hx))
            predicted = outputs.max(1)[1]
            inputs = self.embed(predicted).view(predicted.size(0), -1)
            sampled_ids.append(predicted)

        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids




    def beamSearchBatched(self, features, states, n=3):
        features = self.conv(features)
        features = features.view(features.size(0), -1)
        features = self.bn(self.fc(features))

        inputs = features

        hx = states[0].squeeze()

        hx = self.gru_attention(inputs, hx, features)   # (batch_size, 1, hidden_size)
        outputs = self.linear(hx)                    # (batch_size, vocab_size)
        confidences, best_choices = outputs.topk(n)

        cached_states = [hx] * n

        end_idx = self.vocab.word2idx["<end>"]
        
        #best_choices = best_choices[:,0]
        terminated_choices = []
        terminated_confidences = []


        # one loop for each word
        for word_index in range(50):
            #print(best_choices)
            best_list = [None] * n * (n - len(terminated_choices))
            best_confidence = [None] * n * (n - len(terminated_choices))
            best_states = [None] * n * (n - len(terminated_choices))

            print(best_choices)
            # for each choice
            for choice_index, choice in enumerate(best_choices.transpose(-1,0)):
                # print(choice)
                input_choice = choice[word_index] if word_index != 0 else choice
                # print(input_choice)
                inputs = self.embed(input_choice)
                # print(inputs)
                current_confidence = confidences[:,choice_index]

                hx = self.gru_attention(inputs=inputs,hx=cached_states[choice_index], features=features)        # (batch_size, 1, hidden_size)
                outputs = self.linear(hx)                                                                       # (batch_size, vocab_size)

                # pick n best nodes for each possible choice
                inner_confidences, inner_best_choices = outputs.topk(n)

                #inner_confidences = inner_confidences[0]
                inner_best_choices = inner_best_choices.transpose(-1,0)
                inner_confidences  = inner_confidences.transpose(-1,0)
                print("---- LOOP ----")
                #print(outputs)
                for inner_choice_idx, inner_choice in enumerate(inner_best_choices):
                    print(inner_choice)

                    if word_index != 0:
                        choice = best_choices[choice_index]

                    item = torch.cat((choice, inner_choice))
                    #print(item)


                    position = choice_index * n + inner_choice_idx

                    #print()
                    #print()
                    #print(inner_confidences[inner_choice_idx])
                    #print(current_confidence)
                    #print(best_confidence[position])

                    best_list[position]   = item
                    best_states[position] = hx
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
                #cached_states = [ best_states[index] for index in best_choices_index ]
                confidences   = [ best_confidence_tensor[index] for index in best_choices_index ]
                confidences   = torch.cat(confidences,0).unsqueeze(0)
            else:
                break

    class Node:
        def __init__(choice_list, confidence, states):
            self.choice_list = choice_list
            self.confidence = confidence
            self.states = states

    class NodeList:
        def __init__():
            self.nodes = []

    def beamSearch(self, features, states, n=1, diverse_gamma=0.0):
        features = self.conv(features)
        features = features.view(features.size(0), -1)
        features = self.bn(self.fc(features))

        inputs = features

        states = states[0].squeeze(0)
        states = self.gru_attention(inputs, states, features)   # (batch_size, 1, hidden_size)
        outputs = self.log_softmax(self.linear(states.squeeze(1)))           # (batch_size, vocab_size)
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

            # node_list = [None] * n * (n - len(terminated_choices))

            # for each choice
            for choice_index, choice in enumerate(best_choices):

                input_choice = choice[word_index] if word_index != 0 else choice
                   
                inputs = self.embed(input_choice)
                current_confidence = confidences[:,choice_index]

                out_states = self.gru_attention(inputs, cached_states[choice_index], features) # (batch_size, 1, hidden_size)
                outputs = self.log_softmax(self.linear(out_states.squeeze(1)))                 # (batch_size, vocab_size)

                # pick n best nodes for each possible choice
                inner_confidences, inner_best_choices = outputs.topk(n)
                inner_confidences, inner_best_choices = inner_confidences.squeeze(), inner_best_choices.squeeze()
                for rank, inner_choice in enumerate(inner_best_choices):
                    if word_index != 0:
                        choice = best_choices[choice_index]
                    item = torch.cat((choice, inner_choice))

                    position = choice_index * n + rank

                    confidence = current_confidence + inner_confidences[rank] - (diverse_gamma * (rank + 1))

                    #node_list[position] = Node(choice_list=item, confidence=confidence, states=out_states)

                    best_list[position]   = item
                    best_states[position] = out_states
                    best_confidence[position] = current_confidence + inner_confidences[rank] - (diverse_gamma * (rank + 1))


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


    def beamSearchOld(self, features, states, n=5):
        inputs = features.unsqueeze(1)

        hiddens, states = self.gru(inputs, states)          # (batch_size, 1, hidden_size)
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

class D(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(D, self).__init__()

        gru_embed_size = embed_size
        fc_embed_size = gru_embed_size * 2

        self.gru = nn.GRU(gru_embed_size, gru_embed_size, num_layers,dropout=0.5, batch_first=True, bidirectional=True)


        self.linear = nn.Linear(fc_embed_size, fc_embed_size)
        self.linear2 = nn.Linear(fc_embed_size, fc_embed_size)
        self.linear3 = nn.Linear(fc_embed_size, 2)



        self.fc = nn.Linear(len(vocab), gru_embed_size)
        self.embed = nn.Embedding(len(vocab), gru_embed_size)

        self.fc.weights = self.embed.weight


        #self.fc = nn.Linear(2048, embed_size)

        self.dropout = nn.Dropout(0.5)
        # self.sigmoid = nn.Sigmoid()
        # self.log_softmax = nn.LogSoftmax()

        #self.bn2 = nn.BatchNorm1d(fc_embed_size, momentum=0.01)
        #self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        self.relu = nn.ReLU()

        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        #self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

        self.linear2.weight.data.uniform_(-0.1, 0.1)
        self.linear2.bias.data.fill_(0)

        self.linear3.weight.data.uniform_(-0.1, 0.1)
        self.linear3.bias.data.fill_(0)

        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        #self.fc.weight.data.normal_(0.0, 0.02)
        #self.fc.bias.data.fill_(0)

    # def forward(self, hidden, embeddings,lengths):
    #     """Discriminate feature vectors generated via teacher forcing and free running."""

    #     ## only get embeddings up to the second last word as the last word is not used for teaching
    #     hidden = torch.cat((hidden, embeddings[:,:-1,:]),2)
    #     inputs = pack_padded_sequence(hidden, lengths, batch_first=True )
    #     hiddens, _ = self.gru(inputs)

    #     hiddens = pad_packed_sequence(hiddens, batch_first=True)[0]

    #     x = torch.cat([ hiddens[ i, lengths[i] - 1 ].unsqueeze(0) for i in range( len(lengths) ) ], 0)

    #     x = self.relu(self.bn2(self.linear(x)))
    #     x = self.relu(self.bn2(self.linear2(x)))
    #     x = self.linear3(x)

    #     return self.sigmoid(x)

    def forward(self, inputs,lengths,batch_sizes, label=True):
        """Discriminate feature vectors generated via teacher forcing and free running."""

        ## only get embeddings up to the second last word as the last word is not used for teaching
        #inputs = torch.cat((hidden, embeddings[:,:-1,:]),2)
        #inputs = pack_padded_sequence(hidden, lengths, batch_first=True )
        #hiddens, _ = self.gru(inputs)
        if label:
            inputs = self.embed(inputs)
        else:
            inputs = torch.exp(inputs)
            inputs = self.fc(inputs)

        inputs = PackedSequence(data=inputs, batch_sizes=batch_sizes)

        hiddens, _ = self.gru(inputs)
        hiddens = pad_packed_sequence(hiddens, batch_first=True)[0]

        x = torch.cat([ hiddens[ i, lengths[i] - 1 ].unsqueeze(0) for i in range( len(lengths) ) ], 0)

        x = self.relu(self.linear(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return self.log_softmax(x)


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