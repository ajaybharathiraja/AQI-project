"""Small residual LSTM; independent from custom structures and input adapters."""
import torch
from torch import nn


class PollutantLSTM(nn.Module):
    """One 32-unit unidirectional layer predicts a correction to persistence.

    Input [batch, sequence, features] standardized by TRAIN statistics. Output
    is standardized PM2.5. Hidden state is reset for each independent sequence.
    """
    def __init__(self, features=7, hidden=32, persistence_scale=1., persistence_offset=0.):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer('persistence_scale', torch.tensor(float(persistence_scale)))
        self.register_buffer('persistence_offset', torch.tensor(float(persistence_offset)))

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[2] != self.lstm.input_size:
            raise ValueError('Expected [batch, sequence, trained feature count]')
        _, (hidden, _) = self.lstm(inputs)
        return self.head(hidden[-1]).squeeze(-1) + inputs[:, -1, 0]*self.persistence_scale + self.persistence_offset
