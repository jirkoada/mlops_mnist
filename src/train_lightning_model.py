import torch
import argparse
from pytorch_lightning import Trainer
from models.model_lightning import LightningModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb


sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'val_loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'lr': {
            'values': [1e-2, 1e-3, 1e-4]
        },
        'bs': {
            'values': [32, 64]
        },
        'epochs': {
            'values': [5, 10]
        }
    }
}


def sweep():
    # Initialize a new wandb run
    run = wandb.init()

    # Load hyperparameters from the sweep
    lr = run.config.lr
    bs = run.config.bs
    epochs = run.config.epochs

    train(lr, epochs, bs)


def train(lr, epochs, bs):
    model = LightningModel(lr=lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models/LightningModel/checkpoints", monitor="val_loss", mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="Lightning", log_model='all')
    trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint_callback, early_stopping_callback], logger=wandb_logger)

    train_set = torch.load('data/processed/corruptmnist/train.pt')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)

    test_set = torch.load('data/processed/corruptmnist/test.pt')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False)

    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="Lightning")
    wandb.agent(sweep_id, function=sweep, count=10)
    """parser = argparse.ArgumentParser(description='Train a PyTorch Lightning model.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    train(args.lr, args.epochs, args.bs)"""
