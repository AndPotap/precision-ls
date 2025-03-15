import torch

from .curriculum import Curriculum
from .schema import schema


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
