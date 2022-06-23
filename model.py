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
        x = x.view(x.shape[0], -1 )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # output nodes for each different label
        out1 = torch.sigmoid(self.out1(x))
        out2 = torch.sigmoid(self.out2(x))
        out3 = torch.sigmoid(self.out3(x))
        out4 = torch.sigmoid(self.out4(x))
        out5 = torch.sigmoid(self.out5(x))
        out6 = torch.sigmoid(self.out6(x))

        return out1, out2, out3, out4, out5, out6

# custom Binary loss function

def binary_loss(outputs, targets):
    o1, o2, o3, o4, o5, o6 = outputs
    t1, t2, t3, t4, t5, t6 = targets
    avg1 = nn.BCELoss()(o1, t1)
    avg2 = nn.BCELoss()(o2, t2)
    avg3 = nn.BCELoss()(o3, t3)
    avg4 = nn.BCELoss()(o4, t4)
    avg5 = nn.BCELoss()(o5, t5)
    avg6 = nn.BCELoss()(o6, t6)
    return (avg1+avg2+avg3+avg4+avg5+avg6)/6



# model = MultiClassifier(32, 300)

# data = torch.rand(16, 32, 300)

# output = model.forward(data)
# print(len(output))
