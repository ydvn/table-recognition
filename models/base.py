from abc import abstractmethod

from torch import nn


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self):
        pass
