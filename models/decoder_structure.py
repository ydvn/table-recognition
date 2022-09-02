import torch
from torch import nn
from torchvision import models

from models.base import BaseModel
from models.soft_attention import SoftAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoderStructureWithAttention(BaseModel):
    def __init__(
        self,
        structure_vocab,
        embedding_dim=16,
        encoder_dim=512,
        decoder_dim=256,
        attention_dim=512,
        dropout=0.5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.structure_vocab = structure_vocab
        self.structure_vocab_size = len(structure_vocab)
        self.id2words = {value: key for key, value in structure_vocab.items()}

        # Model components
        self.soft_attention = SoftAttention(
            self.encoder_dim, self.decoder_dim, self.attention_dim
        )  # soft attention network

        self.embedding = nn.Embedding(
            self.structure_vocab_size, self.embedding_dim
        )  # embedding network
        self.dropout = nn.Dropout(p=self.dropout)  # dropout

        self.decoder = nn.LSTMCell(
            self.encoder_dim + self.embedding_dim, self.decoder_dim, bias=True
        )  # decoder LSTMcell
        self.initial_hidden_state = nn.Linear(
            self.encoder_dim, self.decoder_dim
        )  # linear layer to find initial hidden state of LSTMCell
        self.initial_cell_state = nn.Linear(
            self.encoder_dim, self.decoder_dim
        )  # linear layer to find initial cell state of LSTMCell

        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # linear layer to find vocab score
        self.fc = nn.Linear(self.decoder_dim, self.structure_vocab_size)

        self.init_weight()

    def init_weight(self):
        """Intialize some parameters with values from uniform distribution for easier convergence"""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        Args:
            encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)

        Returns:
            hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        hidden_state = self.initial_hidden_state(mean_encoder_out)
        cell_state = self.initial_cell_state(mean_encoder_out)
        return hidden_state, cell_state

    def forward(self, encoder_out, encoded_structure_labels, structure_label_length):
        batch_size = encoder_out.size(0)
        encoder_dim = self.encoder_dim
        vocab_size = self.structure_vocab_size

        # Flatten image
        encoder_out = encoder_out.view(
            batch_size, -1, encoder_dim
        )  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing length
        structure_label_length, sort_idx = structure_label_length.squeeze(1).sort(
            dim=0, descending=True
        )
        encoder_out = encoder_out[sort_idx]
        encoded_structure_labels = encoded_structure_labels[sort_idx]

        # Embedding
        embeddings = self.embedding(
            encoded_structure_labels
        )  # (batch_size, max_structure_label_length, embedding_dim)

        # Initialize LSTM hidden state
        lstm_hidden_state, lstm_cell_state = self.init_hidden_state(
            encoder_out
        )  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (structure_label_length - 1).tolist()

        # Create tensor to hold structure token prediction and alpha
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(
            device
        )
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # create hidden state to generate cell
        hidden_states = [[] for i in range(batch_size)]

        for _curr in range(max(decode_lengths)):
            batch_size_curr = sum([l > _curr for l in decode_lengths])
            attention_weighted_score, alpha = self.soft_attention(
                encoder_out[:batch_size_curr], lstm_hidden_state[:batch_size_curr]
            )
            # gating scalar, (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(lstm_hidden_state[:batch_size_curr]))
            attention_weighted_score = gate * attention_weighted_score

            # hidden h_t+1 and ground_truth token t_t
            lstm_hidden_state, lstm_cell_state = self.decoder(
                torch.cat(
                    [embeddings[:batch_size_curr, _curr, :], attention_weighted_score],
                    dim=1,
                ),
                (
                    lstm_hidden_state[:batch_size_curr],
                    lstm_cell_state[:batch_size_curr],
                ),
            )  # (batch_size_t, decoder_dim)

            # get and save hidden state h_k+1 when groun_truth token in t_k is <td> or >
            for i in range(batch_size_curr):
                if (
                    self.structure_vocab["<td>"]
                    == encoded_structure_labels[i][_curr].cpu().numpy()
                    | self.structure_vocab[">"]
                    == encoded_structure_labels[i][_curr].cpu().numpy()
                ):
                    hidden_states[i].append(lstm_hidden_state[i])

            preds = self.fc(
                self.dropout(lstm_hidden_state)
            )  # (batch_size_t, vocab_size)
            predictions[:batch_size_curr, _curr, :] = preds
            alphas[:batch_size_curr, _curr, :] = alpha
        return (
            predictions,
            encoded_structure_labels,
            decode_lengths,
            alphas,
            hidden_states,
            sort_idx,
        )
