#library imports
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error

""" TOKENIZE AND CREATE A VOCABULARY AND ENCODE SENTENCES
"""

import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "data_first10"

path = "dataset/" + filename + ".csv"
df = pd.read_csv(path)

# tokenization
tok = spacy.load('en_core_web_sm')
def tokenize (text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

#count number of occurences of each word
counts = Counter()
for index, row in df.iterrows():
    counts.update(tokenize(row['content']))


#deleting infrequent words
print("num_words before:",len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
print("num_words after:",len(counts.keys()))


#creating vocabulary
vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)

def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
#     length = min(N, len(enc1))
#     encoded[:length] = enc1[:length]
#     return encoded, length
    return enc1

df['encoded'] = df['content'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))

X = list(df['encoded'])
y = list(df['label'])
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].astype(np.int32)), self.y[idx], len(self.X[idx])

train_ds = ReviewsDataset(X_train, y_train)
val_ds = ReviewsDataset(X_valid, y_valid)


""" DEFINE TRAINING FUNCTIONS
"""

import time

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

import sklearn

def final_evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    y_true = []
    y_pred = []
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            y_pred += list(predited_label.argmax(1).to(torch.device("cpu")))
            y_true += list(label.to(torch.device("cpu")))
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return sklearn.metrics.f1_score(y_true, y_pred), total_acc/total_count, sklearn.metrics.mean_squared_error(y_true, y_pred)


""" SETUP DATALOADER FOR EMBEDDING BAG
"""

from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label, _size) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=collate_batch)
val_dataloader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_batch)

""" SETUP EMBEDDING BAG MODEL
"""

from torch import nn

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

model = TextClassificationModel(len(vocab2index.keys()), 64, 2).to(device)

print("torch available?", torch.cuda.is_available())

""" TEST EMBEDDING BAG MODEL
"""

from torch.utils.data.dataset import random_split

# Hyperparameters
EPOCHS = 100 # epoch
LR = 1  # learning rate
BATCH_SIZE = 8 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, .25, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
total_accu = None

val_accs = []
train_accs = []

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(val_dataloader)
    accu_train = evaluate(train_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    val_accs.append(accu_val)
    train_accs.append(accu_train)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} , train accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val, accu_train))

val_accs_str = [str(x) for x in val_accs]
train_accs_str = [str(x) for x in train_accs]

with open("accs/" + filename + "val_emb_bag.txt", "w") as f:
    f.write("[" + ",".join(val_accs_str) + "]")
with open("accs/" + filename + "train_emb_bag.txt", "w") as f:
    f.write("[" + ",".join(train_accs_str) + "]")


f1, acc, mse = final_evaluate(val_dataloader)
print("F1:", f1, "acc:",  acc, "mse:", mse)
