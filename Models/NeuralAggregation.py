import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
import math


class Aggregation(nn.Module):
    def __init__(self,num_features, num_classes,num_hidden):
        '''
        :param num_features: dimension of input feature, usually 512
        :param num_classes: number of classes (identities) in the dataset
        :param num_hidden: number of neurons in hidden layer
        '''

        super(Aggregation,self).__init__()
        self.q0 = torch.nn.Parameter(torch.zeros(size=(num_features,1)))

        self.intermediate_layer = nn.Linear(num_features,num_features)
        self.tanh = nn.Tanh()
        self.soft0 = nn.Softmax(dim=1)
        self.soft1 = nn.Softmax(dim=1)

        self.fc = nn.Linear(num_features,num_hidden)
        self.prelu = nn.PReLU(num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_classes)


    def forward(self,x):
        batch,frame,feat = x.shape
        x_resized = torch.reshape(x,(batch*frame,feat))

        # first attention
        e0 = torch.matmul(x_resized,self.q0)
        e0 = torch.reshape(e0,(batch,frame))
        a0 = self.soft0(e0)
        a0 = a0.unsqueeze(2).expand_as(x)
        r0 = torch.sum(x * a0, dim=1)

        # second attention
        q1 = self.tanh(self.intermediate_layer(r0))
        q1 = q1.unsqueeze(dim=1)
        e1 = x * q1
        e1 = torch.sum(e1, dim=2)
        a1 = self.soft1(e1)
        a1 = a1.unsqueeze(2).expand_as(x)
        r1 = torch.sum(x * a1, dim=1)

        output = self.fc(r1)
        output = self.fc2(output)
        return output
