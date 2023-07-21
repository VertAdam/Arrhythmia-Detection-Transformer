from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os


from wideAndDeepModel import CTN
from scheduler import NoamScheduler
from dataset import ECG12LeadDataset


def plot_attention_weights(input_data, attention_head = 0):
    # Assume that 'input_data' is your input to the CTN model and 'model' is an instance of the CTN class

    output, attention_weights = model(input_data)

    # Get ECG data
    ecg_data = input_data.detach().cpu().numpy()[0, 0]

    # Prepare figure
    fig, ax1 = plt.subplots()

    # Plot ECG data
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('ECG', color=color)
    ax1.plot(ecg_data, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create second axes that shares the same x-axis
    ax2 = ax1.twinx()

    # We already handled the x-label with ax1
    ax2.set_ylabel('Attention', color=color)

    # Stack attention weights into a tensor
    attention_tensor = torch.stack(attention_weights)

    # Compute attention weights average
    weights_avg = attention_tensor[attention_head].mean(dim=1).mean(dim=-1).detach().cpu().numpy()

    # We apply a moving average to make the attention weights smoother
    # window = 10
    # weights_smooth = np.convolve(weights_avg, np.ones(window)/window, mode='valid')
    from scipy import interpolate

    # Create an interpolation function based on the original weights
    f = interpolate.interp1d(np.arange(len(weights_avg)), weights_avg, kind='linear')

    # Create new x values with the same length as ecg_data
    xnew = np.linspace(0, len(weights_avg) - 1, num=len(ecg_data))

    # Interpolate the weights_avg to these new x values
    weights_avg_interp = f(xnew)
    # Plot attention weights
    color = 'tab:blue'
    ax2.plot(weights_avg_interp, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.set_yscale('log')

    # Show plot
    fig.tight_layout()
    return fig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device Used:', device)
classes_weight = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002',
                  '39732003',
                  '164909002', '251146004', '698252002', '10370003', '284470004', '427172004', '164947007',
                  '111975006',
                  '164917005', '47665007', '59118001', '427393009', '426177001', '426783006', '427084000',
                  '63593006',
                  '164934002', '59931005', '17338001']

classes = sorted(classes_weight)

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

train_dataset = ECG12LeadDataset('data.hdf5', 'x_train', 'y_train')
val_dataset = ECG12LeadDataset('data.hdf5', 'x_val', 'y_val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers = num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers = num_workers)

model = CTN(embedding_size, nhead, feed_fwd_layer_size, n_layers, dropout, fc1_size, wide_feature_size, classes)
model = model.float()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=starting_lr, betas=betas, eps=eps)
scheduler = LambdaLR(optimizer, lr_lambda=NoamScheduler)#lambda step: NoamScheduler(model_size, warmup_steps, step + 1))

writer = SummaryWriter()
run_name = os.path.basename(writer.log_dir)
for epoch in range(n_epochs):
    for batch_idx, batch in enumerate(train_loader):
        batch_features = batch[0].float().to(device)
        batch_labels = batch[1].float().to(device)
        optimizer.zero_grad()
        out, attention_weights = model(batch_features)
        loss = nn.functional.binary_cross_entropy_with_logits(out, batch_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning Rate', current_lr, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch_features = batch[0].float().to(device)
            batch_labels = batch[1].float().to(device)
            out, attention_weights = model(batch_features)
            loss = nn.functional.binary_cross_entropy_with_logits(out, batch_labels)
            val_loss += loss.item()
    writer.add_scalar('Validation Loss', val_loss / len(val_loader), epoch)
    for n in range(8):
        fig = plot_attention_weights(batch_features, n)
        plt.savefig('runs/%s/Epoch %s Attention Head %s.png' % (run_name,epoch, n))
        writer.add_figure('Epoch %s Attention Head %s' % (epoch, n), fig, global_step=epoch * len(train_loader) + batch_idx)
        plt.close(fig)
x = 1

