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
from torch.nn.init import kaiming_uniform

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
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c

        for parameters in self.parameters():
            parameters.requires_grad = requires_grad

    def forward(self, x):
        """Extract the image feature vectors."""
        x = x.clone()
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
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
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)

        return x

class EncoderFC(nn.Module):
    def __init__(self):
        super(EncoderFC, self).__init__()
        self.fc_global = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )


        self.fc_local  = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x_global = F.avg_pool2d(x, kernel_size=x.size()[2:])[:,:,0,0]
        x_global = self.fc_global(x_global)

        # batch x 2048 x 8 x 8 -> batch x 8 x 8 x 2048
        x_local = x.permute(0,2,3,1).contiguous()
        # batch x 8 x 8 x 2048 -> batch * 64 x 2048
        batch, im_size_w , im_size_h , depth = x_local.size()
        x_local = x_local.view( -1 , depth)
        # batch * 64 x 2048 -> batch * 64 x 512
        x_local = self.fc_local(x_local)
        # batch * 64 x 512 -> batch x 64 x 512
        x_local  = x_local.view(batch, im_size_w * im_size_h, -1)

        return x_global, x_local


class G_Spatial(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, num_layers, attn_size, image_first=True):
        """Set the hyper-parameters and build the layers."""
        super(G_Spatial, self).__init__()
        self.encode_fc = EncoderFC()
        self.vocab = vocab

        self.vocab_size = len(vocab)
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.image_first = image_first
        self.start_idx = self.vocab('<start>')
        self.end_idx   = self.vocab('<end>')

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size * 2, self.vocab_size)
        )
        self.embed = nn.Linear(self.vocab_size, embed_size)
        self.v2h = nn.Linear(embed_size * 2, embed_size)

        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size)
        self.attn = nn.Linear( hidden_size, attn_size)

        self.log_softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                kaiming_uniform(m.weight.data)
                m.bias.data.zero_()

    def lstm_attention(self, inputs, hx,cx, features):
        inputs = self.v2h(inputs)
        hx, cx = self.lstm_cell(inputs, (hx,cx))
        attn_weights = F.softmax(self.attn(hx))
        cx = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)
        # skip connection
        #cx = cx + visual_cx
        return hx, cx

    def lstm_attention_classical(self, inputs, hx,cx, features):
        inputs = self.v2h(inputs)
        attn_weights = F.softmax(self.attn(hx)).unsqueeze(1)
        cx = torch.bmm(attn_weights, features).squeeze(1)
        hx, cx = self.lstm_cell(inputs, (hx,cx))
        # skip connection
        #cx = cx + visual_cx
        return hx, cx

         
    def forward(self, features, captions, lengths, state, teacher_forced=True):
        """Decode image feature vectors and generates captions."""
        if teacher_forced:
            return self._forward_forced_cell(features, captions, lengths, state)
        else:
            return self._forward_free_cell(features, lengths, state)

    def _forward_forced_cell(self, features, captions, lengths, states):
        features_global, features_local = features
        if isinstance(captions, tuple):
            captions, batch_sizes = captions
            embeddings = self.embed(captions)
            embeddings = pad_packed_sequence((embeddings,batch_sizes), batch_first=True)[0]
        else:
            batch_size, caption_length, _ = captions.size()
            captions = captions.view(-1,self.vocab_size)
            embeddings = self.embed(captions)
            embeddings = embeddings.view(batch_size, caption_length, -1)

        if self.image_first:
            embeddings = torch.cat((features_global.unsqueeze(1), embeddings), 1)
        else:
            onehot = torch.cuda.FloatTensor(features_global.size(0), self.vocab_size).fill_(0)
            onehot[:,self.start_idx] = 1
            onehot = Variable(onehot)
            inputs = self.embed(onehot)

            embeddings = torch.cat((inputs.unsqueeze(1), embeddings), 1)

        hiddens_ctx_tensor = Variable(torch.cuda.FloatTensor(len(lengths),lengths[0],self.hidden_size * 2))
        hx, cx = states
        hx, cx = hx[0], cx[0]

        batch_size = features_global.size(0)
        if hx.size(0) != batch_size:
            hx = hx[:batch_size]
            cx = cx[:batch_size]

        for i in range(lengths[0]):
            inputs = embeddings[:,i,:]
            inputs = torch.cat((inputs, features_global),1)

            hx, cx = self.lstm_attention(inputs, hx,cx, features_local)
            hiddens_ctx_tensor[ :, i, :] = torch.cat((hx,cx),1)

        hiddens_ctx_tensor_packed, _ = pack_padded_sequence(hiddens_ctx_tensor, lengths, batch_first=True)
        outputs = self.fc(hiddens_ctx_tensor_packed)
        return outputs

    def _forward_free_cell(self, features, lengths, states, adversarial=False):
        output_tensor = Variable(torch.cuda.FloatTensor(len(lengths),lengths[0],self.vocab_size))
        features_global, features_local = features

        if self.image_first:
            inputs = features_global
        else:
            onehot = torch.cuda.FloatTensor(features_global.size(0), self.vocab_size).fill_(0)
            onehot[:,self.start_idx] = 1
            onehot = Variable(onehot)
            inputs = self.embed(onehot)

        hx, cx = states
        hx, cx = hx[0], cx[0]

        batch_size = features_global.size(0)
        if hx.size(0) != batch_size:
            hx = hx[:batch_size]
            cx = cx[:batch_size]

        for i in range(lengths[0]):
            inputs = torch.cat((inputs, features_global),1)
            hx, cx = self.lstm_attention(inputs, hx,cx, features_local)

            combined = torch.cat((hx,cx), 1)
            outputs = self.fc(combined)
            tau = 0.5 if adversarial else 1e-5
            outputs = self.gumbel_sample(outputs, tau=tau)
            inputs = self.embed(outputs).view(outputs.size(0), -1)

            output_tensor [:,i,:] = outputs

        output_tensor, _ = pack_padded_sequence(output_tensor, lengths, batch_first=True)
        #hiddens_tensor, _ = pack_padded_sequence(hiddens_tensor, lengths, batch_first=True)
        return output_tensor

    def gumbel_sample(self,input, tau):
        noise = torch.rand(input.size()).cuda()
        noise.add_(1e-9).log_().neg_()
        noise.add_(1e-9).log_().neg_()
        noise = Variable(noise)
        x = (input + noise) / tau
        x = F.softmax(x.view(input.size(0), -1))
        return x.view_as(input)

    def sample(self, features, states):
        MAX_LENGTH=50
        features_global, features_local = features

        output_tensor = torch.cuda.LongTensor(features_local.size(0),MAX_LENGTH,1)
        features_global, features_local = features
        if self.image_first:
            inputs = features_global
        else:
            onehot = torch.cuda.FloatTensor(features_global.size(0), self.vocab_size).fill_(0)
            onehot[:,self.start_idx] = 1
            onehot = Variable(onehot)
            inputs = self.embed(onehot)

        hx, cx = states
        hx, cx = hx[0], cx[0]

        batch_size = features_global.size(0)
        if hx.size(0) != batch_size:
            hx = hx[:batch_size]
            cx = cx[:batch_size]

        for i in range(MAX_LENGTH):
            inputs = torch.cat((inputs, features_global),1)
            hx, cx = self.lstm_attention(inputs, hx,cx, features_local)

            combined = torch.cat((hx,cx), 1)
            outputs = self.fc(combined)
            predicted = outputs.max(1)[1]
            output_tensor [:,i,:] = predicted.data
            onehot = torch.cuda.FloatTensor(features_global.size(0), self.vocab_size).fill_(0)

            predicted_numpy = predicted.data.cpu().numpy()
            for j in range(64):
                onehot[j, predicted_numpy[j][0]] = 1

            onehot = Variable(onehot)
            inputs = self.embed(onehot)


        return output_tensor

    def beamSearch(self, features, states, n=1, diverse_gamma=0.0):
        features_global, features_local = features

        if self.image_first:
            inputs = Variable(features_global.data, volatile=True)
        else:
            onehot = torch.cuda.FloatTensor(1, self.vocab_size).fill_(0)
            onehot[:,self.start_idx] = 1
            onehot = Variable(onehot, volatile=True)
            inputs = self.embed(onehot)

        inputs = torch.cat((inputs, features_global), 1)

        hx, cx = states
        hx, cx = hx[0], cx[0]
        hx,cx = self.lstm_attention(inputs, hx,cx, features_local)     # (batch_size, 1, hidden_size)
        combined = torch.cat((hx,cx), 1)

        outputs = self.log_softmax(self.fc(combined))             # (batch_size, vocab_size)
        confidences, best_choices = outputs.topk(n)

        cached_states = [hx] * n
        cached_context = [cx] * n
        
        best_choices = best_choices[0]
        terminated_choices = []
        terminated_confidences = []

        # one loop for each word
        for word_index in range(50):
            #print(best_choices)
            best_list = [None] * n * (n - len(terminated_choices))
            best_confidence = [None] * n * (n - len(terminated_choices))
            best_states = [None] * n * (n - len(terminated_choices))
            best_context = [None] * n * (n - len(terminated_choices))

            for choice_index, choice in enumerate(best_choices):
                input_choice = choice[word_index] if word_index != 0 else choice

                onehot = torch.cuda.FloatTensor(1, self.vocab_size).fill_(0)
                onehot[0][input_choice.data] = 1
                onehot = Variable(onehot, volatile=True)

                inputs = self.embed(onehot)
                inputs = torch.cat((inputs, features_global), 1)

                current_confidence = confidences[:,choice_index]
                hx, cx = self.lstm_attention(inputs,
                 cached_states[choice_index], cached_context[choice_index], features_local) # (batch_size, 1, hidden_size)

                combined = torch.cat((hx,cx), 1)
                outputs = self.log_softmax(self.fc(combined))                       # (batch_size, vocab_size)

                # pick n best nodes for each possible choice
                inner_confidences, inner_best_choices = outputs.topk(n)
                inner_confidences, inner_best_choices = inner_confidences.squeeze(), inner_best_choices.squeeze()
                for rank, inner_choice in enumerate(inner_best_choices):
                    if word_index != 0:
                        choice = best_choices[choice_index]
                    item = torch.cat((choice, inner_choice))
                    position = choice_index * n + rank
                    confidence = current_confidence + inner_confidences[rank] - (diverse_gamma * (rank + 1))

                    best_list[position]   = item
                    best_states[position] = hx
                    best_context[position] = cx
                    best_confidence[position] = current_confidence + inner_confidences[rank] - (diverse_gamma * (rank + 1))


            best_confidence_tensor = torch.cat(best_confidence, 0)
            _ , topk = best_confidence_tensor.topk(n - len(terminated_choices))

            topk_index = topk.data.int().cpu().numpy()

            ## Filter nodes that contains termination token '<end>'
            best_choices_index = [ index for index in topk_index if self.end_idx not in best_list[index].data.int() ]

            terminated_index   = list(set(topk_index) - set(best_choices_index))
            terminated_choices.extend([ best_list[index] for index in terminated_index ])
            terminated_confidences.extend([ best_confidence_tensor[index].data.cpu().numpy()[0] for index in terminated_index ])

            ## If there is still nodes to evaluate
            if len(best_choices_index) > 0:
                best_choices  = [ best_list[index] for index in best_choices_index ]
                cached_states = [ best_states[index] for index in best_choices_index ]
                cached_context = [ best_context[index] for index in best_choices_index ]
                confidences   = [ best_confidence_tensor[index] for index in best_choices_index ]
                confidences   = torch.cat(confidences,0).unsqueeze(0)
            else:
                break

        #print(best_choices)
        if len(best_choices_index) > 0:
            terminated_choices.extend([ best_list[index] for index in best_choices_index ])
            terminated_confidences.extend([ best_confidence_tensor[index].data.cpu().numpy()[0] for index in best_choices_index ])

        return terminated_choices,terminated_confidences
