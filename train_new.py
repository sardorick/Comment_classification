import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertModel
from model_bert import Classifier
from data_handler_new import train_loader, test_loader, train_data, test_data
np.random.seed(0)
torch.manual_seed(0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = Classifier(bert_model)
df = pd.read_csv('data/toxic_comments.csv').drop(columns='id')

learning_rate = 0.001
optimizer = optim.Adam(params=model.parameters(), lr = learning_rate)
criterion = nn.BCEWithLogitsLoss()
n_epochs = 5

# Train loop
benchmark_acc = 0.70
train_losses = []
test_losses = []
test_acc = []
for epoch in range(n_epochs):
    running_train_loss = 0
    running_test_loss = 0
    running_test_acc = 0
    
    model.to(device)
    model.train()
    print('############# Epoch {} #############'.format(epoch + 1))
    start = time.time()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        # pack out needed data for the model
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        targets = data['targets'].to(device)

        # forward pass
        outputs = model(ids, mask, token_type_ids)

        # Cost
        loss = criterion(outputs, targets)
        
        # backward
        loss.backward()

        # weights update
        optimizer.step()
        
        
        running_train_loss += loss.item()
    train_losses.append(running_train_loss / len(train_loader))
    
    print(f'TrainLoss: {train_losses[-1] :.4f}')
    
    ######################    
    # validate the model #
    ######################
 
    model.eval()
   
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            targets = data['targets'].to(device)
            outputs = model(ids, mask, token_type_ids)

            loss = criterion(outputs, targets)

            running_test_loss += loss.item()

            # accuracy
            targets = targets.cpu().int().numpy()
            probs = torch.sigmoid(outputs).cpu().detach()
            preds = (probs >= 0.5).int().numpy()

            running_test_acc += accuracy_score(targets, preds)

        test_acc.append(running_test_acc / len(test_loader))
        test_losses.append(running_test_loss / len(test_loader))
        print(f'TestLoss: {test_losses[-1] :.4f}')
        print(f'TestAccu: {test_acc[-1] *100 :.2f}%')
        print(f'Time: {time.time() - start :.4f}')


        # save best model
        if test_acc[-1] > benchmark_acc:
            # save model to cpu
            torch.save(model.to('cpu').state_dict(), './model.pth')
            

            # update benckmark
            benchmark_acc = test_acc[-1]
