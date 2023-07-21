import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from utils import load_data, train_loop, evaluate
from CNN import CNN


def main():

	learning_rate = 0.001
	batch_size = 512
	epochs = 25
	torch.manual_seed(42)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("You are using device: %s" % device)

	model = CNN()
	print(model)

	model = model.to(device)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	train_loader, val_loader, test_loader = load_data('data/mitbih_train.csv',
													  'data/mitbih_test.csv',
													  batch_size,
													  val_split=0.2)

	train_accuracy_curve = []
	val_accuracy_curve = []

	train_loss_curve = []
	val_loss_curve = []

	for t in range(epochs):

		print(f"Epoch {t+1}\n-------------------------------")
		train_accuracy, train_loss = train_loop(train_loader, model, loss_fn, optimizer, device)
		val_accuracy, val_loss = evaluate(val_loader, model, loss_fn, device)
		print(f"Train Accuracy: {train_accuracy:>8f}  Train Loss: {train_loss:>8f}")
		print(f"Val Accuracy: {val_accuracy:>8f}  Val Loss: {val_loss:>8f}")

		train_accuracy_curve.append(train_accuracy)
		val_accuracy_curve.append(val_accuracy)

		train_loss_curve.append(train_loss)
		val_loss_curve.append(val_loss)

	plt.plot(train_accuracy_curve, label='train')
	plt.plot(val_accuracy_curve, label='val')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Accuracy Curve')
	plt.legend()
	plt.savefig('plots/accuracy_curve.png')
	plt.close()

	plt.plot(train_loss_curve, label='train')
	plt.plot(val_loss_curve, label='val')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss Curve')
	plt.legend()
	plt.savefig('plots/loss_curve.png')
	plt.close()

	test_accuracy, _ = evaluate(test_loader, model, loss_fn, device)
	print(f"Test Accuracy: {test_accuracy:>8f}")


if __name__ == "__main__":
    main()

