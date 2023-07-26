import wfdb
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0, 1))

# Then continue with the rest of your processing...

label_dict = {'N':0,'S':1,'V':2,'F':3,'Q':4}
data = []
for id in range(1000):
    n = str(id).zfill(3)
    print(n)
    record_name = 'icentia11k data/p00/p00%s/p00%s_s10'% (n,n)

    try:

        record = wfdb.rdrecord(record_name)
        annotation = wfdb.rdann(record_name, 'atr')
    except FileNotFoundError:
        continue
    # To match dataset the frequency is downsampled from 250Hz to 125Hz
    p_signal = record.p_signal[::2]
    beat_idx = (annotation.sample/2).astype(int)
    symbols = annotation.symbol

    for b in range(100,110):
        beat_start = beat_idx[b-1]
        beat_end = (beat_idx[b]+(beat_idx[b+1]-beat_idx[b])*0.2).astype(int)
        beat = p_signal[beat_start:beat_end,0]
        beat = scaler.fit_transform(beat.reshape(-1, 1)).reshape(-1)
        if len(beat) > 186: # Remove all beats that are longer than 186 samples seconds
            continue
        beat_padded = np.pad(beat, (0, 187 - len(beat)))
        try:
            l = label_dict[symbols[b]]
        except KeyError:
            continue
        data.append(np.append(beat_padded,l))
        x = 1
train, test = train_test_split(np.array(data), test_size = 0.2, random_state = 111)
train_x = train[:,:-1]
train_y = train[:,-1]
test_x = test[:,:-1]
test_y = test[:,-1]

train_x_tensor = torch.from_numpy(train_x).float()
train_y_tensor = torch.from_numpy(train_y).long()
test_x_tensor = torch.from_numpy(test_x).float()
test_y_tensor = torch.from_numpy(test_y).long()

train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
test_dataset = TensorDataset(test_x_tensor, test_y_tensor)

# Save datasets
torch.save(train_dataset, 'icentia11k_train_dataset.pt')
torch.save(test_dataset, 'icentia11k_test_dataset.pt')


x  = 1