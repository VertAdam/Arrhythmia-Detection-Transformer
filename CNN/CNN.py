import torch.nn as nn
import torch.nn.functional as F

# Model based on paper from https://arxiv.org/abs/1805.00794 
# ECG Heartbeat Classification: A Deep Transferable Representation

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1)

        self.conv1_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.conv1_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.conv3_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.conv3_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.conv4_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.conv4_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.conv5_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.conv5_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.pool5 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.conv(x)

        act1_1 = F.relu(self.conv1_1(x))
        act1_2 = F.relu(self.conv1_2(act1_1) + x)
        pool1 = self.pool1(act1_2)

        act2_1 = F.relu(self.conv2_1(pool1))
        act2_2 = F.relu(self.conv2_2(act2_1) + pool1)
        pool2 = self.pool2(act2_2)

        act3_1 = F.relu(self.conv3_1(pool2))
        act3_2 = F.relu(self.conv3_2(act3_1) + pool2)
        pool3 = self.pool3(act3_2)

        act4_1 = F.relu(self.conv4_1(pool3))
        act4_2 = F.relu(self.conv4_2(act4_1) + pool3)
        pool4 = self.pool4(act4_2)

        act5_1 = F.relu(self.conv5_1(pool4))
        act5_2 = F.relu(self.conv5_2(act5_1) + pool4)
        pool5 = self.pool5(act5_2)

        x = self.flatten(pool5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)

        return x

