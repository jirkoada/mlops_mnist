import click
import torch
from torch import nn, optim
from models.model import MyAwesomeModel
import matplotlib.pyplot as plt

from data.make_dataset import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="training epochs")
@click.option("--bs", default=64, help="batch size")
def train(lr, epochs, bs):
    """Train a model on MNIST."""
    
    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set = torch.load('data/processed/corruptmnist/train.pt')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)

    test_set = torch.load('data/processed/corruptmnist/train.pt')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, test_losses = [], []
    step_losses = []
    for e in range(epochs):
        print(f"\nEpoch {e+1}")
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step_losses.append(loss.item())

        print(f"Training loss: {running_loss / len(train_loader)}")
        train_losses.append(running_loss / len(train_loader))
        torch.save(model.state_dict(), f"models/MyAwesomeModel/checkpoints/ep{e+1}.pth")

        with torch.no_grad():
            model.eval()
            val_loss = 0
            correct = 0
            for images, labels in test_loader:
                log_ps = model(images)
                preds = torch.argmax(log_ps, dim=1)
                loss = criterion(log_ps, labels)
                val_loss += loss.item()
                correct += torch.sum(preds == labels)
            print(f"Validation loss: {val_loss / len(test_loader)}")
            acc = correct / len(test_loader.dataset)
            print(f"Accuracy: {acc*100}%")
            test_losses.append(val_loss / len(test_loader))

    plt.figure()
    plt.plot(train_losses, label="Training loss")
    plt.plot(test_losses, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('reports/figures/MyAwesomeModelLosses.png')
    plt.show()

    plt.figure()
    plt.plot(step_losses)
    plt.xlabel("Steps")
    plt.ylabel("Training loss")
    plt.savefig('reports/figures/MyAwesomeModelTLoss.png')
    plt.show()


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    test_set = torch.load('data/processed/corruptmnist/test.pt')
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

    with torch.no_grad():
        model.eval()
        correct = 0
        for images, labels in test_loader:
            log_ps = model(images)
            preds = torch.argmax(log_ps, dim=1)
            correct += torch.sum(preds == labels)
        acc = correct / len(test_loader.dataset) * 100
        print(f'Accuracy on the test set: {round(acc.item(), 4)}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
