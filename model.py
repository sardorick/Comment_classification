import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassifier(nn.Module):
    def __init__(self, max_seq_len, emb_dim) -> None:
        super(MultiClassifier, self).__init__()
        self.fc1 = nn.Linear(max_seq_len*emb_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)

        # Defining output layers for each different label of the classifier
        self.out1 = nn.Linear(256, 1)
        self.out2 = nn.Linear(256, 1)
        self.out3 = nn.Linear(256, 1)
        self.out4 = nn.Linear(256, 1)
        self.out5 = nn.Linear(256, 1)
        self.out6 = nn.Linear(256, 1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # output nodes for each different label
        out1 = F.sigmoid(self.out1(x))
        out2 = F.sigmoid(self.out2(x))
        out3 = F.sigmoid(self.out3(x))
        out4 = F.sigmoid(self.out4(x))
        out5 = F.sigmoid(self.out5(x))
        out6 = F.sigmoid(self.out6(x))

        return out1, out2, out3, out4, out5, out6
    
model = MultiClassifier(32, 300)

data = torch.rand(16, 32, 300)

output = model.forward(data)
print(output)
