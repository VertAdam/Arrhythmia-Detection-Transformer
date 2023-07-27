import optuna
from models import Transformer_no_PE, Transformer, ResNet, ResNetTransformer
import torch
import utils
import torch.nn as nn
import numpy as np


def objective(trial, model_type, epochs = 15):
    # torch.manual_seed(42)
    batch_size = 512#trial.suggest_categorical("batch_size", [256, 512, 1024])
    if model_type =='ResNetTransformer':
        learning_rate = 1e-4 # trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        nchannels = trial.suggest_int("in_channels", 32, 512, step=16)
        in_channels =nchannels
        out_channels = nchannels
        pool_stride = 2
        kernel_size = trial.suggest_int("kernel_size", 3, 7, step = 2)
        num_res_blocks = 5#trial.suggest_int("num_res_blocks", 2, 5)
        # divisors = [i for i in range(1, nchannels + 1) if nchannels % i == 0]
        nhead = trial.suggest_categorical("nhead", [2,4,8,16])
        dim_feedforward = trial.suggest_int("dim_feedforward", 1024, 4096, step=512)
        t_dropout = trial.suggest_float("t_dropout", 0.0, 0.5, step=0.1)
        pe_dropout = trial.suggest_float("pe_dropout", 0.0, 0.5, step=0.1)
        num_layers = trial.suggest_int("num_layers", 3, 9)


        print(f"nchannels: {nchannels}, in_channels: {in_channels}, out_channels: {out_channels}, kernel_size: {kernel_size}, num_res_blocks: {num_res_blocks}, nhead: {nhead}, dim_feedforward: {dim_feedforward}, t_dropout: {t_dropout}, pe_dropout: {pe_dropout}, num_layers: {num_layers}")

        model = ResNetTransformer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                                                    num_res_blocks=num_res_blocks, nhead=nhead,
                                                                    dim_feedforward=dim_feedforward, t_dropout=t_dropout,
                                                                    pe_dropout=pe_dropout, num_layers=num_layers,
                                  pool_stride = pool_stride)
    elif model_type== 'Transformer_no_PE':
        learning_rate = 1e-4#trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        nhead = trial.suggest_categorical("nhead", [2, 4, 8, 10, 20, 40])
        dim_feedforward = trial.suggest_int("dim_feedforward", 1024, 4096, step=512)
        dropout = trial.suggest_float("t_dropout", 0.0, 0.5, step=0.1)
        num_layers = trial.suggest_int("num_layers", 3, 9)
        model = Transformer_no_PE(nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, num_layers=num_layers)
    elif model_type == 'Transformer':
        learning_rate = 1e-4#trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        nhead = trial.suggest_categorical("nhead", [2, 4, 8, 10, 20, 40])
        dim_feedforward = trial.suggest_int("dim_feedforward", 1024, 4096, step=512)
        t_dropout = trial.suggest_float("t_dropout", 0.0, 0.5, step=0.1)
        pe_dropout = trial.suggest_float("pe_dropout", 0.0, 0.5, step=0.1)
        num_layers = trial.suggest_int("num_layers", 3, 9)
        model = Transformer(nhead=nhead, dim_feedforward=dim_feedforward, t_dropout=t_dropout, pe_dropout=pe_dropout,
                            num_layers=num_layers)
    elif model_type =='ResNet':
        learning_rate = 1e-4#trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        nchannels = trial.suggest_int("in_channels", 32, 512, step=16)
        in_channels =nchannels
        out_channels = nchannels
        kernel_size = trial.suggest_int("kernel_size", 3, 7, step = 2)
        num_res_blocks = 5#trial.suggest_int("num_res_blocks", 3, 7)
        model = ResNet(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                       num_res_blocks=num_res_blocks)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader, test_loader = utils.load_data('data/mitbih_train.csv',
                                                            'data/mitbih_test.csv',
                                                            batch_size,
                                                            val_split=0.2)

    train_accuracy_curve = []
    val_accuracy_curve = []

    train_loss_curve = []
    val_loss_curve = []

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_accuracy, train_loss = utils.train_loop(train_loader, model, loss_fn, optimizer, device, verbose = False)
        val_accuracy, val_loss = utils.evaluate(val_loader, model, loss_fn, device)
        print(f"Train Accuracy: {train_accuracy:>8f}  Train Loss: {train_loss:>8f}")
        print(f"Val Accuracy: {val_accuracy:>8f}  Val Loss: {val_loss:>8f}")
        train_accuracy_curve.append(train_accuracy)
        val_accuracy_curve.append(val_accuracy)

        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)

        trial.report(val_accuracy,t)
        # if np.isnan(val_accuracy):
        #     print(val_accuracy)
        #     raise optuna.exceptions.TrialPruned()
    print(f"Val Accuracy: {val_accuracy:>8f}")
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return val_accuracy


if __name__ == "__main__":
    import joblib
    import matplotlib.pyplot as plt
    # Resnet Transformer
    models_types = ['ResNet']# ResNetTransformer, Transformer_no_PE, Transformer
    for model_type in models_types:
        objective_fn = lambda trial: objective(trial, model_type=model_type, epochs=10)
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective_fn, n_trials=50)
        print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
        joblib.dump(study, '%s_optunaStudy.pkl' % model_type)
        fig = optuna.visualization.plot_contour(study)
        fig.write_html("plots/contour_%s.html" % model_type)

        ax = optuna.visualization.matplotlib.plot_param_importances(study)
        ax.set_title('Param Importance %s' % model_type)
        fig = ax.figure
        fig.set_size_inches((15, 10))
        plt.tight_layout()
        fig.savefig("plots/paramImportance_%s.png" % model_type)
        plt.close('all')

        ax = optuna.visualization.matplotlib.plot_intermediate_values(study)
        ax.set_title('All Trials %s' % model_type)
        ax.get_legend().remove()
        fig = ax.figure
        fig.set_size_inches((15, 10))
        plt.tight_layout()
        fig.savefig("plots/allTrials_%s.png" % model_type)
        plt.close('all')
        x= 1
