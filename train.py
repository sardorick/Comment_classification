import torch
import torch.nn as nn
from model import MultiClassifier
from model import binary_loss as criterion
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processer import load_data, train_test_split, train_loader, test_loader

emb_dim = 300
MAX_SEQ_LEN = 32
model = MultiClassifier(MAX_SEQ_LEN, emb_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
num_epochs =  1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# #dummy train loader and test loader
# train_loader = [1,2,3,4,5]
# test_loader = [1,2,3,4,5]

# df = load_data('data/train.csv')
# train_df, test_df = train_test_split(df)


# train loop
all_train_losses, all_test_losses, all_accuracies = [], [], []
print_every = 40
for epoch in range(num_epochs):
    running_loss_train, running_loss_test, running_test_accuracy = 0, 0, 0

    for i, (sentences, labels) in enumerate(iter(train_loader)):
        print(f'Epoch: {epoch+1}, iteration: {i}')
        # sentences.resize_(sentences.size()[0], 32*emb_dim)
        features = sentences.to(device)
        target1 = labels[:, 0].to(device)
        target2 = labels[:, 1].to(device)
        target3 = labels[:, 2].to(device)
        target4 = labels[:, 3].to(device)
        target5 = labels[:, 4].to(device)
        target6 = labels[:, 5].to(device)

        optimizer.zero_grad()
        outputs = model(features)
        targets = target1, target2, target3, target4, target5, target6
        train_loss = criterion(outputs, targets)
        train_loss.backward()
        optimizer.step()
        running_loss_train += train_loss.item()

    avg_running_loss = running_loss_train/len(train_loader)
    all_train_losses.append(avg_running_loss)

    # model.eval()
    # with torch.no_grad():
    #     for i, (sentences_test, labels_test) in enumerate(iter(test_loader)):
    #         # extract the features and labels
    #         features = test_df['comment'].to(device)
    #         target1 = test_df['label1'].to(device)
    #         target2 = test_df['severe_toxic'].to(device)
    #         target3 = test_df['obscene'].to(device)
    #         target4 = test_df['threat'].to(device)
    #         target5 = test_df['insult'].to(device)
    #         target6 = test_df['identity_hate'].to(device)
    #         outputs_test = model(features)
                    
    #         # get all the labels
    #         all_labels = []
    #         for out in outputs_test:
    #             if out >= 0.5:
    #                 all_labels.append(1)
    #             else:
    #                 all_labels.append(0)
            
    #         targets = (target1, target2, target3, target4, target5)
    #         test_loss = criterion(outputs_test, targets)
    #         running_loss_test += test_loss.item()
            
    #         # get all the targets in int format from tensor format
    #         all_targets = []
    #         for target in targets:
    #             all_targets.append(int(target.squeeze(0).detach().cpu()))
                    
    #         print(f"ALL PREDICTIONS: {all_labels}")
    #         print(f"GROUND TRUTHS: {all_targets}")

    #     avg_running_test = running_loss_test/len(test_loader)
    #     all_test_losses.append(avg_running_test)
    print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_running_loss:.4f}") # , Test Loss: {avg_running_test:.4f}

# torch.save(model.state_dict(), 'final_model.pth')
# # plot and save the train loss graph
# plt.figure(figsize=(10, 7))
# plt.plot(all_train_losses, color='orange', label='train loss')
# plt.plot(all_test_losses, label='test loss')

# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('figures.png')
# plt.show()
