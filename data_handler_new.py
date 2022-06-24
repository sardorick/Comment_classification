import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

np.random.seed(0)
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('data/toxic_comments.csv').drop(columns='id')
word_counts = df.comment_text.apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 6))
sns.histplot(x = word_counts)

sample = df.sample(500).reset_index(drop=True)


max_len = 36
batch_size = 32
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# token ids needed to retrieve embeddings
inputs = tokenizer.encode_plus(
            sample.comment_text[0],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

class ToxCOmmentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.df = df
        self.comments = df.comment_text
        self.targets = df.drop(columns='comment_text').values.tolist()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        comment = self.comments[index]
        targets = self.targets[index]

        inputs = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"] 

        return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(targets, dtype=torch.float)
            }

train_data, test_data = train_test_split(sample, test_size=0.2)
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)
train_data.shape, test_data.shape

train_set = ToxCOmmentDataset(train_data, tokenizer, max_len)
test_set = ToxCOmmentDataset(test_data, tokenizer, max_len)

train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size, shuffle=False, drop_last=True)

# print(len(train_loader))
# iter(train_loader).next()['ids'].shape, iter(train_loader).next()['targets'].shape