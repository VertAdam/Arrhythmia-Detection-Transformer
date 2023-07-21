from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from wideAndDeepModel import CTN
from scheduler import NoamScheduler
from dataset import ECG12LeadDataset

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

for epoch in range(n_epochs):
    for batch_idx, batch in enumerate(train_loader):
        batch_features = batch[0].float().to(device)
        batch_labels = batch[1].float().to(device)
        optimizer.zero_grad()
        out = model(batch_features)
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
            out = model(batch_features)
            loss = nn.functional.binary_cross_entropy_with_logits(out, batch_labels)
            val_loss += loss.item()
    writer.add_scalar('Validation Loss', val_loss / len(val_loader), epoch)
x = 1