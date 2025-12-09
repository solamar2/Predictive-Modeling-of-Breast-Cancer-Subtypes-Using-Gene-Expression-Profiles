import torch.nn as nn
from scripts.Constants import numoflabels,dropoutparm

class createANNmodel(nn.Module):
    def __init__(self,numofgenes):
        super().__init__()
        # Define layers
        self.layer1 = nn.Linear(numofgenes, 64)
        self.layer2 = nn.Linear(64,32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, numoflabels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropoutparm)

    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        #x = self.dropout(x)
        x = self.relu(self.layer2(x))
        #x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.layer4(x)
        return x