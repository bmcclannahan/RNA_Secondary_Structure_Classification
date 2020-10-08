import torch
from torch import nn

class EuclidianDistance(nn.Module):
    def __init__(self):
        super(EuclidianDistance, self).__init__()

    def forward(self, x1, x2):
        return torch.abs(x1-x2)

class SiameseNetwork(nn.Module):
    def __init__(self,model_func):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = self.create_cnn(model_func)

        # Defining the fully connected layers
        self.flatten = nn.Sequential(
        # First Dense Layer
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True))
        self.euclidean = EuclidianDistance()
        self.fc = nn.Sequential(
            nn.Linear(1024,1)#,
            #nn.Sigmoid()
        )
    
    def create_cnn(self,model_func):
        modules=list(model_func().children())[:-1]
        cnn = nn.Sequential(*modules)
        for p in cnn.parameters():
            p.requires_grad = False
        return cnn

    def forward_once(self, x):
        # Forward pass 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.flatten(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        left = self.forward_once(input1)
        # forward pass of input 2
        right = self.forward_once(input2)
        # pass both through euclidean layer
        #print('left:',left.data[0][:5])
        #print('right:',right.data[0][:5])
        euclidean = self.euclidean(left,right)
        #print('euclidean:',euclidean.data[0][:5])
        output = self.fc(euclidean)
        return output