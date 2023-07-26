import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Transformer_no_PE(nn.Module):

    def __init__(self, d_model=200, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=6, num_classes=5):
        super(Transformer_no_PE, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc1 = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc1(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model=200, pe_dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(pe_dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self, d_model=200, nhead=8, dim_feedforward=2048, t_dropout=0.1, pe_dropout=0.1, num_layers=6, num_classes=5):
        super(Transformer, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, pe_dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, t_dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc1 = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(0)
        x = self.fc1(x)
        return x


# ResNet based on paper from https://arxiv.org/abs/1805.00794 
# ECG Heartbeat Classification: A Deep Transferable Representation


class ResidualBlock(nn.Module):

    def __init__(self, in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same', pool_kernel_size=5, pool_stride=2):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool1d(pool_kernel_size, pool_stride)

    def forward(self, x):
        a = F.relu(self.conv1(x))
        x = F.relu(self.conv2(a) + x)
        x = self.pool(x)
        return x


class ResNet(nn.Module):

    def __init__(self, in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same', pool_kernel_size=5, pool_stride=2, num_res_blocks=5,
                 dim_fc=32, num_classes=5):
        super(ResNet, self).__init__()

        self.conv = nn.Conv1d(1, out_channels, kernel_size, stride)

        self.residual_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            block = ResidualBlock(in_channels, out_channels, kernel_size, stride, padding, pool_kernel_size, pool_stride)
            self.residual_blocks.append(block)

        self.flatten = nn.Flatten()

        with torch.no_grad():
            d = torch.randn(1, 200)
            d = self.conv(d)
            for block in self.residual_blocks:
                d = block(d)
            dim_in = d.size(0) * d.size(1)

        self.fc1 = nn.Linear(dim_in, dim_fc)
        self.fc2 = nn.Linear(dim_fc, dim_fc)
        self.fc3 = nn.Linear(dim_fc, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.conv(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class ResNetTransformer(nn.Module):

    def __init__(self, in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same', pool_kernel_size=5, pool_stride=2, num_res_blocks=5,
                 d_model=96, nhead=8, dim_feedforward=2048, t_dropout=0.1, pe_dropout=0.1, num_layers=6, num_classes=5):
        super(ResNetTransformer, self).__init__()

        self.conv = nn.Conv1d(1, out_channels, kernel_size, stride)

        self.residual_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            block = ResidualBlock(in_channels, out_channels, kernel_size, stride, padding, pool_kernel_size, pool_stride)
            self.residual_blocks.append(block)

        self.flatten = nn.Flatten()

        with torch.no_grad():
            d = torch.randn(1, 200)
            d = self.conv(d)
            for block in self.residual_blocks:
                d = block(d)
            d_model = d.size(0) * d.size(1)

        self.pos_encoder = PositionalEncoding(d_model, pe_dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, t_dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc1 = nn.Linear(d_model, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.conv(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.flatten(x)
        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(0)
        x = self.fc1(x)

        return x
