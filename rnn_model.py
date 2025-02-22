import torch
from torch import nn

# Define model


class RNN(nn.Module):
    def __init__(self, tokens_count):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(tokens_count*2, 24),
            nn.ReLU(),
            nn.Linear(24, tokens_count)
        )
        self.softmax = nn.Softmax(dim=0)
        self.last_predictions = torch.zeros(
            (tokens_count), dtype=torch.float32)
        self.tokens_count = tokens_count

    def reset(self):
        self.last_predictions = torch.zeros(
            (self.tokens_count), dtype=torch.float32)

    def detach(self):
        self.last_predictions = self.last_predictions.detach()

    def forward(self, x):
        inp = torch.cat((x, self.last_predictions))
        logits = self.linear_relu_stack(inp)
        predictions = self.softmax(logits)
        self.last_predictions = predictions
        return predictions
