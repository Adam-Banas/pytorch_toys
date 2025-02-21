import torch
from torch import nn

# Define model


class LSTM(nn.Module):
    def __init__(self, tokens_count):
        super().__init__()
        input_size = tokens_count*2
        self.prediction_layer = nn.Sequential(
            nn.Linear(input_size, tokens_count),
            nn.Sigmoid()
        )
        self.ignoring_layer = nn.Sequential(
            nn.Linear(input_size, tokens_count),
            nn.Sigmoid()
        )
        self.forgetting_layer = nn.Sequential(
            nn.Linear(input_size, tokens_count),
            nn.Sigmoid()
        )
        self.selecting_layer = nn.Sequential(
            nn.Linear(input_size, tokens_count),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=0)
        self.with_memory_activation = nn.Sigmoid()
        self.last_predictions = torch.zeros(
            (tokens_count), dtype=torch.float32)
        self.memory = torch.zeros((tokens_count), dtype=torch.float32)
        self.tokens_count = tokens_count

    def reset(self):
        self.last_predictions = torch.zeros(
            (self.tokens_count), dtype=torch.float32)
        self.memory = torch.zeros((self.tokens_count), dtype=torch.float32)

    def forward(self, x):
        inp = torch.cat((x, self.last_predictions))

        # Predict
        predicted = self.prediction_layer(inp)

        # And ignore
        ignoring = self.ignoring_layer(inp)
        activation = predicted * ignoring

        # Combine with memory
        forgetting = self.forgetting_layer(inp)
        memory = forgetting * self.memory
        activation_with_memory = self.with_memory_activation(
            activation + memory)

        # Update memory for the next iteration
        # TODO: So if the memory is detached from the graph, how can the network learn what to remember?
        #       I think a different approach is needed here.
        self.memory = activation_with_memory.detach().clone()

        # Select what part of memory should be returned
        selection = self.selecting_layer(inp)
        logits = selection * activation

        # And go through softmax, to get probabilities
        predictions = self.softmax(logits)
        self.last_predictions = predictions.detach()
        return predictions
