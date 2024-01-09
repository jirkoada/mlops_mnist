import pytest
import torch
from tests import _PROJECT_ROOT
import sys
sys.path.insert(0, _PROJECT_ROOT)
from src.models.model import MyAwesomeModel


@pytest.mark.parametrize("input", [torch.randn((1, 28, 28)), torch.full((1, 28, 28), 0.5)])
def test_model(input):
    model = MyAwesomeModel()
    y = model(input)
    assert y.shape == torch.Size([1, 10]), "Output shape is incorrect"

def test_wrong_dim():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to be a 3D tensor'):
        model(torch.randn(1,2,3,4))

def test_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match=r'Expected sample shape \[1, 28, 28\]'):
        model(torch.randn(1,2,3))
