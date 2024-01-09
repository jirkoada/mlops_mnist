from tests import _TEST_ROOT
import sys
sys.path.append(_TEST_ROOT)
from src.train_model import train


def test_training():
    train(0.01, 1, 64)