import pandas as pd
import numpy as np
import torch
import spacy
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import FastText
np.random.seed(0)
nlp = spacy.load('en_core_web_sm')

# load the dataset


def load_data(pth):
    df = pd.read_csv(pth)
    df.drop(columns=['id'], inplace=True)
    return df

# split it to train and test


def train_test_split(df, train_size=0.8):
    df_idx = [i for i in range(len(df))]
    np.random.shuffle(df_idx)
    len_train = int(len(df) * train_size)
    df_train = df.iloc[:len_train].reset_index(drop=True)
    df_test = df.iloc[len_train:].reset_index(drop=True)
    return df_train, df_test

# preprocess


def preprocessing(sentence):
    """
    Clean texts, get rid of symbols, lower case the words, spaces
    return token list
    """
    doc = nlp(sentence)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space and not token.is_punct and not token.is_bracket and not
              token.like_email and not token.is_currency and not token.is_digit and not token.like_url and token.is_ascii]
    return tokens


def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0


def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]


def padding(list_of_indexes, max_seq_len, padding_index=1):
    output = list_of_indexes + \
        (max_seq_len - len(list_of_indexes)) * [padding_index]
    return output[:max_seq_len]


class CommentDataset(Dataset):
    def __init__(self, df, max_seq_len=32):
        self.max_seq_len = max_seq_len
        train_iter = iter(df.comment_text.values)
        self.vec = FastText("simple")

        self.vec.vectors[1] = -torch.ones(self.vec.vectors[1].shape[0])

        self.vec.vectors[0] = torch.zeros(self.vec.vectors[0].shape[0])

        self.vectorizer = lambda x: self.vec.vectors[x]

        # features in form of indices
        sequences = [padding(encoder(preprocessing(
            sequence), self.vec), max_seq_len) for sequence in df.comment_text.tolist()]
        self.sequences = sequences

        # target
        self.labels = df.drop(columns='comment_text').values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        features = self.sequences[index]
        labels = self.labels[index]

        label1 = torch.tensor(labels[0], dtype=torch.float32)
        label2 = torch.tensor(labels[1], dtype=torch.float32)
        label3 = torch.tensor(labels[2], dtype=torch.float32)
        label4 = torch.tensor(labels[3], dtype=torch.float32)
        label5 = torch.tensor(labels[4], dtype=torch.float32)
        label6 = torch.tensor(labels[5], dtype=torch.float32)
        return {
            'comment': features,
            'toxic': label1,
            'severe_toxic': label2,
            'obscene': label3,
            'threat': label4,
            'insult': label5,
            'indentity_hate': label6
        }
