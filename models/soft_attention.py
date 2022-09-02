import torch
from torch import nn

from models.base import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoftAttention(BaseModel):
    def __init__(self, encoder_dim, decoder_dim, attention_dim=512):
        """Self attention layer of LSTM

        Args:
            encoder_dim (int): Feature size of encoded images
            decoder_dim (int): Size of Decoder's RNN
            attention_dim (int, optional): Size of attention network. Defaults to 512.
        """
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        encoder_att = self.encoder_att(
            encoder_out
        )  # (batch_size, num_pixels, attention_dim)
        decoder_att = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)

        # attention is created by output encoder and previous decoder output
        # size (batch_size, num_pixels)
        full_att = self.full_att(
            self.relu(encoder_att + decoder_att.unsqueeze(1))
        ).squeeze(
            2
        )  # size is (batch_size, num_pixels)
        alpha = self.softmax(full_att)  # size is (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_att * alpha.unsqueeze(2)).sum(
            dim=1
        )  # size is (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha
