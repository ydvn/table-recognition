import torch
from torch import nn
from torchvision import models

from models.base import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoderCellWithAttention(BaseModel):
    def __init__(self):
        super().__init__()

    def forward():
        pass
