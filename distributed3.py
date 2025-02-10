import collections
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import tqdm
import os 
import time
start_time = time.time()
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Data Preparation
train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])
tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    length = len(tokens)
    return {"tokens": tokens, "length": length}

max_length = 256
train_data = train_data.map(tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})
test_data = test_data.map(tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})

train_valid_data = train_data.train_test_split(test_size=0.25)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]

min_freq = 5
special_tokens = ["<unk>", "<pad>"]
vocab = torchtext.vocab.build_vocab_from_iterator(train_data["tokens"], min_freq=min_freq, specials=special_tokens)
unk_index = vocab["<unk>"]
pad_index = vocab["<pad>"]
vocab.set_default_index(unk_index)

def numericalize_example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}

train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})

train_data = train_data.with_format(type="torch", columns=["ids", "label", "length"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label", "length"])
test_data = test_data.with_format(type="torch", columns=["ids", "label", "length"])

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
        batch_length = [i["length"] for i in batch]
        batch_length = torch.tensor(batch_length)
        batch_label = [i["label"] for i in batch]
        batch_label = torch.tensor(batch_label)
        return {"ids": batch_ids, "length": batch_length, "label": batch_label}
    return collate_fn

def get_distributed_data_loader(dataset, batch_size, pad_index, shuffle=False):
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    collate_fn = get_collate_fn(pad_index)
    return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        length = length.cpu().long()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param)

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def train(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    vocab_size = len(vocab)
    embedding_dim = 300
    hidden_dim = 300
    output_dim = len(train_data.unique("label"))
    n_layers = 2
    bidirectional = True
    dropout_rate = 0.5

    model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, pad_index).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.apply(initialize_weights)

    glove_path = '/kaggle/input/glove300/glove.6B.300d.txt'
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    pretrained_embedding = torch.zeros(len(vocab), 300)
    for i, token in enumerate(vocab.get_itos()):
        vector = embeddings_index.get(token)
        if vector is not None:
            pretrained_embedding[i] = torch.tensor(vector)
    model.module.embedding.weight.data.copy_(pretrained_embedding)

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    train_loader = get_distributed_data_loader(train_data, batch_size=512, pad_index=pad_index, shuffle=True)
    valid_loader = get_distributed_data_loader(valid_data, batch_size=512, pad_index=pad_index)

    best_valid_loss = float("inf")
    for epoch in range(10):
        model.train()
        train_loss = 0
        for batch in tqdm.tqdm(train_loader, desc="training..."):
            ids, length, label = batch["ids"].to(rank), batch["length"], batch["label"].to(rank)
            optimizer.zero_grad()
            predictions = model(ids, length)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss, valid_acc = evaluate(valid_loader, model, criterion, rank)
        print(f"Rank {rank}, Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution Time after data distribution: {execution_time} seconds")

