import torch
import pytest
import os.path
from tests import _PATH_DATA

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    print(f'{_PATH_DATA}/processed/corruptmnist/train.pt')
    train_set = torch.load(f'{_PATH_DATA}/processed/corruptmnist/train.pt')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

    test_set = torch.load(f'{_PATH_DATA}/processed/corruptmnist/test.pt')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    
    labels = set()
    assert len(train_loader) == 30000, "Train dataset did not have the correct number of samples"
    assert len(test_loader) == 5000, "Test dataset did not have the correct number of samples"
    for data, target in train_loader:
        assert data.shape == torch.Size([1, 28, 28]), "A train data sample did not have the correct shape"
        assert target.shape == torch.Size([1]), "A train data label did not have the correct shape"
        labels.add(target.item())
    assert labels == set(range(10)), "Missing labels in the train dataset"
    for data, target in test_loader:
        assert data.shape == torch.Size([1, 28, 28]), "A test data sample did not have the correct shape"
        assert target.shape == torch.Size([1]), "A test data label did not have the correct shape"
