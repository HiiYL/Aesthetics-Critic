import torch
from torch.autograd import Variable

y = torch.cuda.FloatTensor(1,3,299,299)
y = Variable(y)

from models import InvertedCNN

encoder = InvertedCNN(300)

out = encoder(y)
print(out)