import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
import utils
import models
from scheduler import NoamScheduler
from torch.optim.lr_scheduler import LambdaLR
from focal_loss import FocalLoss, reweight
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
import time


def get_predictions_and_labels(loader, model, device):
	model.eval()
	all_preds = []
	all_labels = []

	with torch.no_grad():
		for inputs, labels in loader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)  # get the predicted classes
			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())

	return all_preds, all_labels


def main(learning_rate=0.0001, batch_size=512, epochs=50, model_type='ResNetTransformer', plot = True, loss = 'FocalLoss', tensor_board = False):
	if tensor_board:
		writer = SummaryWriter('runs/' + model_type + '_' + '_' + time.strftime("%Y%m%d-%H%M%S"))
	torch.manual_seed(42)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("You are using device: %s" % device)

	if model_type == 'Transformer_no_PE':
		model = models.Transformer_no_PE(nhead = 10, dim_feedforward=2048, dropout = 0.1, num_layers = 9)

	if model_type == 'Transformer':
		model = models.Transformer(nhead= 40, dim_feedforward= 2048, t_dropout= 0.0, pe_dropout= 0.0, num_layers= 6)

	if model_type == 'ResNet':
		model = models.ResNet(in_channels=512, out_channels=512, kernel_size=7)

	if model_type == 'ResNetTransformer':
		model = models.ResNetTransformer(in_channels=480, out_channels=480, kernel_size = 7, nhead=8, dim_feedforward= 4096, t_dropout=0.1, pe_dropout=0.0, num_layers = 3)

	print(model)

	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = LambdaLR(optimizer,
						 lr_lambda=NoamScheduler)  # lambda step: NoamScheduler(model_size, warmup_steps, step + 1))

	train_loader, val_loader, test_loader = utils.load_data('data/mitbih_train.csv',
															'data/mitbih_test.csv',
															batch_size,
															val_split=0.2)
	if loss == 'CrossEntropy':
		loss_fn = nn.CrossEntropyLoss()
	elif loss =='FocalLoss':
		class_counts = [0, 0, 0, 0, 0]
		for batch, (X, y) in enumerate(train_loader):
			for n in range(5):
				class_counts[n] += len(y[y == n])
		per_cls_weights = reweight(class_counts, beta= 0.9999).to(device)
		loss_fn = FocalLoss(weight=per_cls_weights, gamma = 1)

	train_f1_curve = []
	val_f1_curve = []

	train_loss_curve = []
	val_loss_curve = []
	patience = 30
	best_val_f1 = 0.0
	epochs_no_improve = 0
	for t in range(epochs):

		print(f"Epoch {t+1}\n-------------------------------")
		train_f1, train_loss = utils.train_loop(train_loader, model, loss_fn, optimizer, device, scheduler)
		val_f1, val_loss = utils.evaluate(val_loader, model, loss_fn, device)
		print(f"Train F1 Score: {train_f1:>8f}  Train Loss: {train_loss:>8f}")
		print(f"Val F1 Score: {val_f1:>8f}  Val Loss: {val_loss:>8f}")

		train_f1_curve.append(train_f1)
		val_f1_curve.append(val_f1)
		train_loss_curve.append(train_loss)
		val_loss_curve.append(val_loss)
		preds, labels = get_predictions_and_labels(val_loader, model, device)
		report = classification_report(labels, preds, output_dict=True, digits = 4)
		if tensor_board:
			writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], t)
			f1_scores = {}
			precision_scores = {}
			recall_scores = {}
			writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, t)
			writer.add_scalars('F1 Score', {'train': train_f1, 'val': val_f1}, t)
			for class_label, metrics in report.items():  # check if the label is a number (class index)
				if class_label == 'accuracy':
					continue
				f1_scores[class_label] = metrics['f1-score']
				precision_scores[class_label] = metrics['precision']
				recall_scores[class_label] = metrics['recall']
			writer.add_scalars('F1_score', f1_scores, t)
			writer.add_scalars('Precision', precision_scores, t)
			writer.add_scalars('Recall', recall_scores, t)
			print(classification_report(labels, preds, digits = 4))

		if val_f1 > best_val_f1:
			best_val_f1 = val_f1
			epochs_no_improve = 0
			torch.save(model.state_dict(), 'models/%s_best.pt' % model_type)
		else:
			epochs_no_improve += 1
			if epochs_no_improve >= patience:
				print("Early stopping at epoch %d" % t)
				break


		x = 1

	writer.close()
	model.load_state_dict(torch.load('models/%s_best.pt' % model_type))

	if plot:
		plt.plot(train_f1_curve[:-patience], label='train')
		plt.plot(val_f1_curve[:-patience], label='val')
		plt.xlabel('Epochs')
		plt.ylabel('F1 Score')
		plt.title(f'{model.__class__.__name__}')
		plt.legend()
		plt.savefig(f'plots/{model.__class__.__name__}_f1_curve.png')
		plt.close()

		plt.plot(train_loss_curve[:-patience], label='train')
		plt.plot(val_loss_curve[:-patience], label='val')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title(f'{model.__class__.__name__}')
		plt.legend()
		plt.savefig(f'plots/{model.__class__.__name__}_loss_curve.png')
		plt.close()
		torch.save(model.state_dict(), 'models/%s.pt' % model_type)

		preds, labels = get_predictions_and_labels(val_loader, model, device)
		report = classification_report(labels, preds, digits = 4)
		print(f'Best {model.__class__.__name__}: Validation')
		print(report)

		x = 1

if __name__ == "__main__":
	for m_type in ['ResNetTransformer', 'ResNet', 'Transformer', 'Transformer_no_PE']:
		main(model_type=m_type, epochs = 1500, learning_rate = 5e-2, loss = 'FocalLoss', tensor_board=True, plot=True)
		x= 1
