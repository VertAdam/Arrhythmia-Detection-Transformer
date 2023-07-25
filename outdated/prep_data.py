"""
Code extraction loop adapted from: Automatic Detection and Classification of 12-lead ECGs Using a Deep Neural by
Wenxiao Jia, Xian Xu, Xiao Xu, Yuyao Sun and Xiaoshuang Liu
https://ieeexplore.ieee.org/document/9344409
"""

from sklearn.model_selection import train_test_split
import os
import numpy as np
import h5py
from scipy.io import loadmat

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data

input_directory = 'training'
input_files = []
header_files = []

train_directory = input_directory
for f in os.listdir(train_directory):
    if os.path.isfile(os.path.join(train_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
            'mat'):
        g = f.replace('.mat', '.hea')
        input_files.append(f)
        header_files.append(g)

# the 27 scored classes
classes_weight = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002',
                  '39732003',
                  '164909002', '251146004', '698252002', '10370003', '284470004', '427172004', '164947007',
                  '111975006',
                  '164917005', '47665007', '59118001', '427393009', '426177001', '426783006', '427084000',
                  '63593006',
                  '164934002', '59931005', '17338001']

classes_name = sorted(classes_weight)

num_files = len(input_files)
num_class = len(classes_name)

# initilize the array
set_length = 5000
data_num = np.zeros((num_files, 12, set_length))

data_external = np.zeros((num_files, 2))
classes_num = np.zeros((num_files, num_class))

for cnt, f in enumerate(input_files):
    classes = set()
    tmp_input_file = os.path.join(train_directory, f)
    data, header_data = driver.load_challenge_data(tmp_input_file)

    for lines in header_data:
        if lines.startswith('# Dx'):
            tmp = lines.split(': ')[1].split(',')
            for c in tmp:
                classes.add(c.strip())

        if lines.startswith('# Age'):
            age = lines.split(': ')[1].strip()
            if age == 'NaN':
                age = '60'
        if lines.startswith('# Sex'):
            sex = lines.split(': ')[1].strip()

    for j in classes:
        if j in classes_name:
            class_index = classes_name.index(j)
            classes_num[cnt, class_index] = 1

    data_external[cnt, 0] = float(age) / 100
    data_external[cnt, 1] = np.array(sex == 'Male').astype(int)
    if cnt%100 ==0:
        print("Sample %s out of %s (%s)" % (cnt, len(input_files), cnt/len(input_files)))
    if data.shape[1] >= set_length:
        data_num[cnt, :, :] = data[:, : set_length] / 30000
    else:
        length = data.shape[1]
        data_num[cnt, :, :length] = data / 30000
        # split the training set and testing set
x_train, x_val, x_train_external, x_val_external, y_train, y_val = train_test_split(data_num, data_external,
                                                                                    classes_num, test_size=0.01,
                                                                                    random_state=2020)


# Open the file in write mode
with h5py.File('data.hdf5', 'w') as f:
    # Save each split to a different dataset
    f.create_dataset('x_train', data=x_train)
    f.create_dataset('x_val', data=x_val)
    f.create_dataset('y_train', data=y_train)
    f.create_dataset('y_val', data=y_val)
