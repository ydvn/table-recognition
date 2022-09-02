import torch
from torch import nn
from torchvision import models

from models.base import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderResnet18(BaseModel):
    def __init__(self, encoded_image_size=448, pretrained=True) -> None:
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)  # pre-trained imagenet resnet18

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )
        self.finetune()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward propogation

        Args:
            images (torch.Tensor): images, a tensor of size (batch, 3, image_size, image_size)

        Returns:
            torch.Tensor: Encoded image
        """
        out = self.encoder(
            images
        )  # (batch_size, 512, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 512)
        out = out.permute(0, 2, 3, 1)
        return out

    def finetune(self, finetune=True):
        """Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        Args:
            finetune (bool, optional): True, if allowed finetuning. Defaults to True.
        """
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        for conv in list(self.encoder.children())[5:]:
            for parameter in conv.parameters():
                parameter.requires_grad = finetune
