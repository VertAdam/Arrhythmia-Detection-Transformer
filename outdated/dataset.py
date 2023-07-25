import h5py
import torch
from torch.utils.data import Dataset

class ECG12LeadDataset(Dataset):
    def __init__(self, hdf5_file, x_name, y_name):
        self.h5file = h5py.File(hdf5_file, 'r')
        self.x = self.h5file[x_name]
        self.y = self.h5file[y_name]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


if __name__== "__main__":
    train_dataset = ECG12LeadDataset('data.hdf5', 'x_train', 'y_train')
    val_dataset = ECG12LeadDataset('data.hdf5', 'x_val', 'y_val')
    x= 1