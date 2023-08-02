import torch
import numpy as np
import models
import utils
from main import get_predictions_and_labels
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.gridspec import GridSpec


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

labels_row_sums = [f'{label} ({sum})' for label, sum in zip(classes, row_sums)]
labels_col_sums = [f'{label} ({sum})' for label, sum in zip(classes, col_sums)]
cmNorm = confusion_matrix(y_true, y_pred, normalize = 'true')

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


# Combined

fig = plt.figure(figsize=(15, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])  # we add an extra space at the end for the colorbar

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
cax = fig.add_subplot(gs[0, 2])  # this is where the colorbar will be placed

# Your confusion matrix calculations for the first model here

model.load_state_dict(torch.load('models/ResNetTransformer_best.pt'))

y_pred, y_true = get_predictions_and_labels(test_loader, model, device)

cm = confusion_matrix(y_true, y_pred)
row_sums = cm.sum(axis=1)
col_sums = cm.sum(axis=0)

labels_row_sums = [f'{label} ({sum})' for label, sum in zip(classes, row_sums)]
labels_col_sums = [f'{label} ({sum})' for label, sum in zip(classes, col_sums)]
cmNorm = confusion_matrix(y_true, y_pred, normalize = 'true')

disp = ConfusionMatrixDisplay(confusion_matrix=cmNorm, display_labels=labels_row_sums)
disp.plot(cmap='Blues', ax=ax1, colorbar=False)  # no colorbar for this subplot
ax1.set_xticks(labels=labels_col_sums, ticks = new_locations)
ax1.set_title("A. Class Balanced Focal Loss")

# Your confusion matrix calculations for the second model here

model.load_state_dict(torch.load('models/ResNetTransformer_best_CrossEntropy.pt'))

y_pred, y_true = get_predictions_and_labels(test_loader, model, device)

cm = confusion_matrix(y_true, y_pred)
row_sums = cm.sum(axis=1)
col_sums = cm.sum(axis=0)

labels_row_sums = [f'{label} ({sum})' for label, sum in zip(classes, row_sums)]
labels_col_sums = [f'{label} ({sum})' for label, sum in zip(classes, col_sums)]
cmNorm = confusion_matrix(y_true, y_pred, normalize = 'true')

disp = ConfusionMatrixDisplay(confusion_matrix=cmNorm, display_labels=labels_row_sums)
disp.plot(cmap='Blues', ax=ax2, colorbar=False)  # no colorbar for this subplot
ax2.set_xticks(labels=labels_col_sums, ticks = new_locations)
ax2.set_title("B. Cross Entropy Loss")

# Now we add a colorbar to cax
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1))  # you may need to adjust vmin and vmax
fig.colorbar(sm, cax=cax, orientation='vertical')
fig.tight_layout()
plt.savefig('plots/report_figs/confusion_matrices_normed_combined.png')
x = 1