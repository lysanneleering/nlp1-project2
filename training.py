"""
Course implementations copied from nlp-illc/nlp1-student/part2/experiment.py.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class LabelledCorpus(Dataset):
    def __init__(self, X, y):
        """Receives a list of variable-length token-id sequences and labels."""
        self.X = [torch.tensor(x, dtype=torch.long) for x in X]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_labelled_texts(batched_pairs, padding_value=0):
    X_rows, y_rows = zip(*batched_pairs)
    X_batch = pad_sequence(X_rows, batch_first=True, padding_value=padding_value)
    y_batch = torch.stack(y_rows)
    return X_batch, y_batch

def training_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    if isinstance(device, str):
        device = torch.device(device)

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logprobs = model(X_batch)
        loss = criterion(logprobs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        predictions = logprobs.argmax(dim=1)
        total_correct += (predictions == y_batch).sum().item()
        total_examples += y_batch.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }

def evaluate(model, dataloader, criterion, device="cpu"):
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    preds_list = []
    labels_list = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logprobs = model(X_batch)
            loss = criterion(logprobs, y_batch)

            total_loss += loss.item() * y_batch.size(0)
            predictions = logprobs.argmax(dim=1)
            total_correct += (predictions == y_batch).sum().item()
            total_examples += y_batch.size(0)
            preds_list.append(predictions.tolist())
            labels_list.append(y_batch.tolist())

    return {
        "predictions": preds_list,
        "labels": labels_list,
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }
