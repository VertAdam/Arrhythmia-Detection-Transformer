import torch
import numpy as np
import models
import utils
from main import get_predictions_and_labels
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)

model = models.ResNetTransformer(in_channels=480, out_channels=480, kernel_size=7, nhead=8, dim_feedforward=4096, t_dropout=0.1, pe_dropout=0.0, num_layers=3)

model.to(device)

model.load_state_dict(torch.load('models/ResNetTransformer_best.pt'))

_, _, test_loader = utils.load_data('data/mitbih_train.csv', 'data/mitbih_test.csv', batch_size=512, val_split=0.2)

y_pred, y_true = get_predictions_and_labels(test_loader, model, device)

print(f'Best {model.__class__.__name__}: Test')
print(classification_report(y_true, y_pred, digits=4))

classes = ['N','S','V','F','Q']
counts = np.unique(y_true, return_counts=True)[1]

cm = confusion_matrix(y_true, y_pred)

row_sums = cm.sum(axis=1)
col_sums = cm.sum(axis=0)

# append sums to labels
labels_row_sums = [f'{label} ({sum})' for label, sum in zip(classes, row_sums)]
labels_col_sums = [f'{label} ({sum})' for label, sum in zip(classes, col_sums)]
cmNorm = confusion_matrix(y_true, y_pred, normalize = 'true')
# plot with column sums
disp = ConfusionMatrixDisplay(confusion_matrix=cmNorm, display_labels=labels_row_sums)
disp.plot(cmap='Blues')
new_locations = np.arange(len(labels_col_sums))
plt.xticks(labels=labels_col_sums, ticks = new_locations)
plt.suptitle("Class Balanced Focal Loss")
plt.savefig(f'plots/report_figs/confusion_matrix_focalLoss_normed.png')
plt.close()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)

model = models.ResNetTransformer(in_channels=480, out_channels=480, kernel_size=7, nhead=8, dim_feedforward=4096, t_dropout=0.1, pe_dropout=0.0, num_layers=3)

model.to(device)

model.load_state_dict(torch.load('models/ResNetTransformer_best_CrossEntropy.pt'))

_, _, test_loader = utils.load_data('data/mitbih_train.csv', 'data/mitbih_test.csv', batch_size=512, val_split=0.2)

y_pred, y_true = get_predictions_and_labels(test_loader, model, device)

print(f'Best {model.__class__.__name__}: Test')
print(classification_report(y_true, y_pred, digits=4))

classes = ['N','S','V','F','Q']
counts = np.unique(y_true, return_counts=True)[1]

cm = confusion_matrix(y_true, y_pred)

row_sums = cm.sum(axis=1)
col_sums = cm.sum(axis=0)

# append sums to labels
labels_row_sums = [f'{label} ({sum})' for label, sum in zip(classes, row_sums)]
labels_col_sums = [f'{label} ({sum})' for label, sum in zip(classes, col_sums)]
cmNorm = confusion_matrix(y_true, y_pred, normalize = 'true')
# plot with column sums
disp = ConfusionMatrixDisplay(confusion_matrix=cmNorm, display_labels=labels_row_sums)
disp.plot(cmap='Blues')
new_locations = np.arange(len(labels_col_sums))
plt.xticks(labels=labels_col_sums, ticks = new_locations)
plt.suptitle("Cross Entropy Loss")
plt.savefig(f'plots/report_figs/confusion_matrix_CrossEntropy_normed.png')
plt.close()