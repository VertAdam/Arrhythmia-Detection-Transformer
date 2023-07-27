import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
import utils
import models
from scheduler import NoamScheduler
from torch.optim.lr_scheduler import LambdaLR

def main(learning_rate=0.0001, batch_size=512, epochs=50, model_type='ResNetTransformer', plot = True):

	torch.manual_seed(42)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("You are using device: %s" % device)

	if model_type == 'Transformer_no_PE':
		model = models.Transformer_no_PE(nhead = 10, dim_feedforward=2048, dropout = 0.1, num_layers = 9)

	if model_type == 'Transformer':
		model = models.Transformer(nhead= 40, dim_feedforward= 2048, t_dropout= 0.0, pe_dropout= 0.0, num_layers= 6)

	if model_type == 'ResNet':
		model = models.ResNet()

	if model_type == 'ResNetTransformer':
		model = models.ResNetTransformer()
	
	print(model)

	model = model.to(device)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = LambdaLR(optimizer,
						 lr_lambda=NoamScheduler)  # lambda step: NoamScheduler(model_size, warmup_steps, step + 1))

	train_loader, val_loader, test_loader = utils.load_data('data/mitbih_train.csv',
															'data/mitbih_test.csv',
															batch_size,
															val_split=0.2)

	train_accuracy_curve = []
	val_accuracy_curve = []

	train_loss_curve = []
	val_loss_curve = []

	for t in range(epochs):

		print(f"Epoch {t+1}\n-------------------------------")
		train_accuracy, train_loss = utils.train_loop(train_loader, model, loss_fn, optimizer, device, scheduler)
		val_accuracy, val_loss = utils.evaluate(val_loader, model, loss_fn, device)
		print(f"Train Accuracy: {train_accuracy:>8f}  Train Loss: {train_loss:>8f}")
		print(f"Val Accuracy: {val_accuracy:>8f}  Val Loss: {val_loss:>8f}")

		train_accuracy_curve.append(train_accuracy)
		val_accuracy_curve.append(val_accuracy)

		train_loss_curve.append(train_loss)
		val_loss_curve.append(val_loss)

	test_accuracy, _ = utils.evaluate(test_loader, model, loss_fn, device)
	print(f"Test Accuracy: {test_accuracy:>8f}")
	if plot:
		plt.plot(train_accuracy_curve, label='train')
		plt.plot(val_accuracy_curve, label='val')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.title(f'{model.__class__.__name__} \n train={train_accuracy:.4f} val={val_accuracy:.4f} test={test_accuracy:.4f}')
		plt.legend()
		plt.savefig(f'plots/{model.__class__.__name__}_accuracy_curve.png')
		plt.close()

		plt.plot(train_loss_curve, label='train')
		plt.plot(val_loss_curve, label='val')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title(f'{model.__class__.__name__} \n train={train_loss:.4f} val={val_loss:.4f}')
		plt.legend()
		plt.savefig(f'plots/{model.__class__.__name__}_loss_curve.png')
		plt.close()

if __name__ == "__main__":
	main(model_type='Transformer', epochs = 50, learning_rate = 1e-2)
	x= 1
