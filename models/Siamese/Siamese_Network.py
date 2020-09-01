from torch import nn

class SiameseNetwork(nn.Module):
    def __init__(self,model_func):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = self.create_cnn(model_func)

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
        # First Dense Layer
        nn.Linear(30976, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.5),
        # Second Dense Layer
        nn.Linear(1024, 128),
        nn.ReLU(inplace=True),
        # Final Dense Layer
        nn.Linear(128,2))

    
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
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        # returning the feature vectors of two inputs
        return output1, output2