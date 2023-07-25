from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from wideAndDeepModel_SingleBeat import CTN_SingleBeat
from scheduler import NoamScheduler
# from dataset import ECG12LeadDataset


def plot_attention_weights(input_data):

    output, attention_weights = model(input_data)

    ecg_data = input_data.detach().cpu().numpy()[0, 0]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('ECG', color=color)
    ax1.plot(ecg_data, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()

    ax2.set_ylabel('Attention', color=color)

    attention_to_plot = attention_weights[-1].to('cpu').detach().unsqueeze(0).numpy()
    attention_to_plot = attention_to_plot.mean(axis=1)[0][0]#

    from scipy import interpolate

    f = interpolate.interp1d(np.arange(len(attention_to_plot)), attention_to_plot, kind='linear')

    xnew = np.linspace(0, len(attention_to_plot) - 1, num=len(ecg_data))

    weights_avg_interp = f(xnew)
    color = 'tab:blue'
    ax2.plot(weights_avg_interp, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    return fig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device Used:', device)

# Loader
batch_size = 128
num_workers = 0

# Model
embedding_size = 256
nhead = 8
feed_fwd_layer_size = 2048
n_layers = 8
dropout = 0.2
fc1_size = 64
wide_feature_size = 0

# Optimizer
starting_lr = 1e-3
betas = (0.9,0.98)
eps = 1e-9
model_size = 12
warmup_steps = 4000

# Training
n_epochs = 100
early_stop_count = None # TODO: Implement early stopping

# train_dataset = ECG12LeadDataset('data.hdf5', 'x_train', 'y_train')
# val_dataset = ECG12LeadDataset('data.hdf5', 'x_val', 'y_val')
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers = num_workers)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers = num_workers)
from CNN.utils import load_data
train_loader, val_loader, test_loader = load_data(r'single_beat_data/mitbih_train.csv',r'single_beat_data/mitbih_test.csv', batch_size)

model = CTN_SingleBeat(embedding_size, nhead, feed_fwd_layer_size, n_layers, dropout, fc1_size, wide_feature_size, [0,1,2,3,4], num_leads = 1)
model = model.float()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=starting_lr, betas=betas, eps=eps)
scheduler = LambdaLR(optimizer, lr_lambda=NoamScheduler)#lambda step: NoamScheduler(model_size, warmup_steps, step + 1))

writer = SummaryWriter()
run_name = os.path.basename(writer.log_dir)
for epoch in range(n_epochs):
    y_true_train = []
    y_pred_train = []
    for batch_idx, batch in enumerate(train_loader):
        batch_features = batch[0].float().to(device)
        batch_labels = batch[1].float().to(device).int()
        batch_labels = torch.eye(5).to(device)[batch_labels]
        optimizer.zero_grad()
        batch_features = batch_features.unsqueeze(1)
        out, attention_weights = model(batch_features)
        loss = nn.functional.cross_entropy(out, batch_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        _, predicted = torch.max(out.data, 1)
        y_true_train.extend(torch.max(batch_labels, 1)[1].cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('SINGLE BEAT Learning Rate', current_lr, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('SINGLE BEAT Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
    precision_train = precision_score(y_true_train, y_pred_train, average='macro')
    recall_train = recall_score(y_true_train, y_pred_train, average='macro')
    f1_train = f1_score(y_true_train, y_pred_train, average='macro')
    acc_val = accuracy_score(y_true_train, y_pred_train)

    writer.add_scalar('SINGLE BEAT Training Accuracy', acc_val, epoch)
    writer.add_scalar('SINGLE BEAT Training Precision', precision_train, epoch)
    writer.add_scalar('SINGLE BEAT Training Recall', recall_train, epoch)
    writer.add_scalar('SINGLE BEAT Training F1', f1_train, epoch)

    # Validation loop
    model.eval()
    val_loss = 0
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch_features = batch[0].float().to(device)
            batch_labels = batch[1].float().to(device).int()
            batch_features = batch_features.unsqueeze(1)
            batch_labels = torch.eye(5).to(device)[batch_labels]
            out, attention_weights = model(batch_features)
            loss = nn.functional.binary_cross_entropy_with_logits(out, batch_labels)
            val_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            y_true_val.extend(torch.max(batch_labels, 1)[1].cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())
    precision_val = precision_score(y_true_val, y_pred_val, average='macro')
    recall_val = recall_score(y_true_val, y_pred_val, average='macro')
    f1_val = f1_score(y_true_val, y_pred_val, average='macro')
    acc_val = accuracy_score(y_true_val, y_pred_val)

    writer.add_scalar('SINGLE BEAT Validation Accuracy', acc_val, epoch)
    writer.add_scalar('SINGLE BEAT Validation Precision', precision_val, epoch)
    writer.add_scalar('SINGLE BEAT Validation Recall', recall_val, epoch)
    writer.add_scalar('SINGLE BEAT Validation F1', f1_val, epoch)
    writer.add_scalar('SINGLE BEAT Validation Loss', val_loss / len(val_loader), epoch)
    # fig = plot_attention_weights(batch_features)
    # plt.savefig('runs/%s/SINGLE BEAT Epoch %s Attention Weights.png' % (run_name,epoch))
    # writer.add_figure('SINGLE BEAT  Epoch %s Attention Weights' % epoch, fig, global_step=epoch * len(train_loader) + batch_idx)
    # plt.close(fig)
x = 1

