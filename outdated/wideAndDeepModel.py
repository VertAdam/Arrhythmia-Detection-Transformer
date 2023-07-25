"""
Model and Code adapted from: A Wide and Deep Transformer Neural Network for 12-Lead ECG Classification
https://ieeexplore.ieee.org/document/9344053
"""

import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add position encodings to embeddings
        # x: embedding vects, [B x L x d_model]
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, nhead, feed_fwd_layer_size, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embedding_size, nhead)
        self.linear1 = nn.Linear(embedding_size, feed_fwd_layer_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feed_fwd_layer_size, embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention layer
        src2, self.attn_weights = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward layer
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, self.attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        output = src
        attention_weights = []
        for mod in self.layers:
            output, weights = mod(output)
            attention_weights.append(weights)

        return self.norm(output), attention_weights


class TransformerFull(nn.Module):
    '''
    Transformer encoder processes convolved ECG samples
    Stacks a number of TransformerEncoderLayers
    '''

    def __init__(self, embedding_size, nhead, feed_fwd_layer_size, num_layers, dropout):
        super(TransformerFull, self).__init__()
        self.embedding_size = embedding_size
        self.nhead = nhead
        self.feed_fwd_layer_size = feed_fwd_layer_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.pe = PositionalEncoding(embedding_size, dropout=dropout)

        # encode_layer = TransformerEncoderLayer(
        #     embedding_size=self.embedding_size,
        #     nhead=self.nhead,
        #     feed_fwd_layer_size=self.feed_fwd_layer_size,
        #     num_layers=self.num_layers,
        #     dropout=self.dropout)
        self.transformer_encoder = TransformerEncoder(embedding_size, nhead, feed_fwd_layer_size, num_layers, dropout)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.pe(out)
        out = out.permute(1, 0, 2)
        out, attention_weights = self.transformer_encoder(out)
        out = out.mean(0)  # global pooling
        return out, attention_weights


# 15 second model
class CTN(nn.Module):
    def __init__(self, embedding_size, nhead, feed_fwd_layer_size, num_layers, dropout, fc1_size, wide_feature_size, classes, num_leads = 12):
        super(CTN, self).__init__()

        self.encoder = nn.Sequential(  # downsampling factor = 20
            nn.Conv1d(num_leads, 128, kernel_size=14, stride=3, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, embedding_size, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_size, embedding_size, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_size, embedding_size, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_size, embedding_size, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True)
        )
        self.transformer = TransformerFull(embedding_size, nhead, feed_fwd_layer_size, num_layers, dropout=dropout)
        self.fc1 = nn.Linear(embedding_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size + wide_feature_size, len(classes))
        self.dropout = nn.Dropout(dropout)

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)

    def forward(self, x, wide_feats = None):
        z = self.encoder(x)  # encoded sequence is batch_sz x nb_ch x seq_len
        out, attention_weights = self.transformer(z)  # transformer output is batch_sz x d_model
        out = self.dropout(F.relu(self.fc1(out)))
        if wide_feats is None:
            out = self.fc2(out)
        else:
            out = self.fc2(torch.cat([wide_feats, out], dim=1))
        return out, attention_weights
