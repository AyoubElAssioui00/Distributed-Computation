import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
import datasets
import numpy as np
import torch.nn as nn
import tqdm
import os
import time

start_time = time.time()
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])
tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    return {"tokens": tokens}

max_length = 256
train_data = train_data.map(tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})
test_data = test_data.map(tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})

train_valid_data = train_data.train_test_split(test_size=0.25)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]

min_freq = 5
special_tokens = ["<unk>", "<pad>"]
vocab = torchtext.vocab.build_vocab_from_iterator(train_data["tokens"],min_freq=min_freq,specials=special_tokens,)
unk_index = vocab["<unk>"]
pad_index = vocab["<pad>"]
vocab.set_default_index(unk_index)

def numericalize_example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}

train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})

train_data = train_data.with_format(type="torch", columns=["ids", "label"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch

    return collate_fn

def get_distributed_data_loader(dataset, batch_size, pad_index, shuffle=False):
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    collate_fn = get_collate_fn(pad_index)
    return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

class CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout_rate,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, n_filters, filter_size)
                for filter_size in filter_sizes
            ]
        )
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids):
        # ids = [batch size, seq len]
        embedded = self.dropout(self.embedding(ids))
        # embedded = [batch size, seq len, embedding dim]
        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch size, embedding dim, seq len]
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n filters, seq len - filter_sizes[n] + 1]
        pooled = [conv.max(dim=-1).values for conv in conved]
        # pooled_n = [batch size, n filters]
        cat = self.dropout(torch.cat(pooled, dim=-1))
        # cat = [batch size, n filters * len(filter_sizes)]
        prediction = self.fc(cat)
        # prediction = [batch size, output dim]
        return prediction

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy
def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
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
    n_filters = 100
    filter_sizes = [3, 5, 7]
    output_dim = len(train_data.unique("label"))
    dropout_rate = 0.25
    model = CNN(
    vocab_size,
    embedding_dim,
    n_filters,
    filter_sizes,
    output_dim,
    dropout_rate,
    pad_index,
    ).to(device)
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
            ids = batch["ids"].to(rank)
            label = batch["label"].to(rank)
            optimizer.zero_grad()
            prediction = model(ids)
            loss = criterion(prediction, label)
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






"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
criterion = criterion.to(device)

def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        )
        
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)





n_epochs = 10
best_valid_loss = float("inf")

metrics = collections.defaultdict(list)

for epoch in range(n_epochs):
    train_loss, train_acc = train(
        train_data_loader, model, criterion, optimizer, device
    )
    
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "cnn.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")"""

