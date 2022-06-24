import pandas as pd
import numpy as np
import torch
import spacy
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import FastText
np.random.seed(0)
BATCH_SIZE = 16

nlp = spacy.load('en_core_web_sm')
fasttext = FastText("simple")
# load the dataset


def load_data(pth):
    df = pd.read_csv(pth)
    df = df.sample(100).reset_index(drop=True)
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

        return features, labels

###################


df = load_data("data/train.csv")
# print(df.head())
train_df, test_df = train_test_split(df)
# print(len(train_df), len(test_df))


train_data, test_data = CommentDataset(
    train_df), CommentDataset(test_df)

# print(test_data[0])


def collate_train(batch, vectorizer=train_data.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token)
                         for token in sentence[0]]) for sentence in batch])

    target = torch.LongTensor([item[1] for item in batch])
    return inputs, target


def collate_test(batch, vectorizer=test_data.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token)
                         for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor([item[1] for item in batch])
    return inputs, target


train_loader = DataLoader(train_df, batch_size=BATCH_SIZE,
                          collate_fn=collate_train, shuffle=True, drop_last=True)


test_loader = DataLoader(
    test_df, batch_size=BATCH_SIZE, collate_fn=collate_test, drop_last=True)


inputs, targets = iter(train_loader).next()
print(inputs.shape)

