"""
3 course implementations of classes copied from nlp-illc/nlp1-student/part2/clslib.py.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class EmbedAndPoolEncoder(nn.Module):
    """
    This encoder embeds a sequence of token ids and then pools the result along sequence length
    using sum or mean.
    """
    def __init__(self, embedding_matrix, average_pool=True, padding_idx=0, freeze_embeddings=True):
        """
        embedding_matrix: a torch tensor of shape (vocab_size, emb_dim)
        average_pool: use True for mean and False for sum
        padding_idx: the id of the pad token (this is for batched computation)
            suppose we are batching three sequences [1, 2, 3], [6, 2] and [2, 1, 4, 5]
            as they differ in length, we use the following trick to batch them into a single tensor
            [
                [1, 2, 3, 0],
                [6, 2, 0, 0],
                [2, 1, 4, 5]
            ]
            that is we extend the shorter sequences with 0s until they match the longest sequence
        freeze_embeddings: tells torch that we are not updating the parameters in the embedding matrix
        """
        super().__init__()
        self.average_pool = average_pool
        # Pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=padding_idx
        )

    @property
    def enc_dim(self):
        return self.embedding.embedding_dim

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        """
        Encodes each sequence in the batch as a single vector of size emb_dim.

        x: token ids with shape (batch_size, max_seq_length, emb_dim)
        """
        # [..., max_seq_length, emb_dim]
        e = self.embedding(x)
        # Normalization for pooling
        if self.average_pool:
            # Total non-padding tokens (min of 1 to avoid problems when averaging the sum of embs)
            # [..., max_seq_length, 1]  # keepdim=True keeps the last axis, this is needed for division later on
            den = torch.sum(x.ne(self.embedding.padding_idx), dim=-1, keepdim=True).clamp_min(1)
        else:  # sum pooling
            den = 1
        # [..., emb_dim]
        h = torch.sum(e, -2) / den  # see that we reduce-sum the sequence dimension, not the representation dimension
        return h


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_matrix, padding_idx=0, freeze_embeddings=True, word_dropout=0.):
        """
        hidden_size: size of the recurrent state
        embedding_matrix: word vectors
        padding_idx: id of the pad token
        freeze_embeddings: treat embeddings as frozen
        """
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        # Pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=padding_idx
        )
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1, # this could be turned into an argument
            batch_first=True  # important, in NLP we treat the first axis of the tensor as the batch dimension, and the second as the sequence dimension
        )

    @property
    def enc_dim(self):
        return self.hidden_size

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        """
        Encode each sequence in the batch into a single H-dimensional vector.

        x: a batch of sequences of token ids
        """
        # [...,max_seq_len, emb_dim]
        embedded = self.embedding(x)
        valid = x.ne(self.embedding.padding_idx)

        # Word dropout is a regularisation technique for recurrent architectures
        # If you are curious why we have to drop words in order to regularise RNNs,
        # here's the paper that figured it out: https://proceedings.neurips.cc/paper_files/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
        # In a nutshell, if we apply dropout before linear layers inside the RNN cell
        # as sequences grow long, information is completely destroyed.
        # That paper builds on a previous paper that interpreted dropout as a specific kind of approximate inference (for a specific kind of approximate Bayesian model)
        # and then establishes that to achieve the same in a recurrent architecture, one can drop words as we do here
        # (By dropping a word, we mean 0-out its embedding)
        if self.training and self.word_dropout > 0:
            # Independently zero each non-padding token with probability word_dropout.
            # [..., max_seq_len]
            drop_mask = (
                torch.rand(x.shape, device=embedded.device) < self.word_dropout # at a certain pre-specified rate
            ) & valid
            # we 0-out the embedding of a valid (not pad) position
            embedded = embedded.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        # [..., max_seq_len] this is necessary for packing
        lengths = torch.sum(valid, dim=-1)
        # for efficiency, we let torch do the for loop, and for that we need to 'pack' our sentences
        # using a torch helper
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False) # sorting leads to even more efficient processing, but it is cumbersome and we skip it
        output, (hidden, cell) = self.lstm(packed)
        final_hidden = hidden[-1]
        return final_hidden


class SoftmaxRegressor(nn.Module):
    """
    This is a softmax regressor,
     it works with encoded text h(x) as input.
    The map from h(x) to outputs uses an MLP.

    One important detail: for numerical stability the output is not softmax-activitate,
     rather, it is logsoftmax-activated. This means the outputs are log prob masses, not prob masses.
    """
    def __init__(self, hidden_size1, hidden_size2, num_classes, encoder: nn.Module, dropout=0.):
        """
        hidden_size1: size of first hidden layer (in the output transformation)
        hidden_size2: size of second hidden layer
        num_classes: number of possible classes
        encoder: an nn.Module that maps each x to a single vector of size encoder.enc_dim
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout), # dropout is a regularisation technique (whereby at some rate, we drop/0-out inputs to a linear layer)
            nn.Linear(encoder.enc_dim, hidden_size1),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size2, num_classes),
            nn.LogSoftmax(dim=-1) # for numerical stability, it is common to use log softmax, rather than softmax
        )

    def forward(self, x):
        # [..., enc_dim]
        h = self.encoder(x)
        # [...,num_classes]
        logprobs = self.output_layer(h)
        return logprobs
