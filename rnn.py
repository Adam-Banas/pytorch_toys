import random
import copy

import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

tokens = ['Jane', 'John', 'dog', 'saw', '.', '\n']
token_to_index = {token: idx for idx, token in enumerate(tokens)}

# Configuration - read from command line
def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--load_model_path", type=str, default=None, help="A path from which to load the model. Will skip training phase if given.")
    parser.add_argument("-s", "--save_model_path", type=str, default=None, help="A path to which save the model")

    return parser.parse_args()

args = parse_args()
load_model_path = args.load_model_path
save_model_path = args.save_model_path

# Input
input_data = [['Jane', 'saw', 'John', '.', '\n'],
['Jane', 'saw', 'dog', '.', '\n'],
['John', 'saw', 'Jane', '.', '\n'],
['John', 'saw', 'dog', '.', '\n'],
['dog', 'saw', 'Jane', '.', '\n'],
['dog', 'saw', 'John', '.', '\n']]

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(tokens)*2, 24),
            nn.ReLU(),
            nn.Linear(24, len(tokens))
        )
        self.softmax = nn.Softmax(dim=0)
        self.last_predictions = torch.zeros((len(tokens)), dtype=torch.float32)

    
    def reset(self):
        self.last_predictions = torch.zeros((len(tokens)), dtype=torch.float32)


    def forward(self, x):
        inp = torch.cat((x, self.last_predictions))
        logits = self.linear_relu_stack(inp)
        predictions = self.softmax(logits)
        self.last_predictions = predictions.detach()
        return predictions

model = NeuralNetwork().to(device)
print(model)

# Utility functions
def prepare_dataset(data):
    shuffled = copy.deepcopy(data)
    random.shuffle(shuffled)
    tokens = []
    for sentence in shuffled:
        for word in sentence:
            tokens.append(word)
    
    X = tokens[:-1]
    y = tokens[1:]
    return X, y

def token_to_one_hot_tensor(token):
    index = token_to_index[token]
    return nn.functional.one_hot(torch.tensor(index), num_classes=len(tokens))

# Train
def train(data, model, loss_fn, optimizer):
    X_set, y_set = prepare_dataset(data)
    size = len(X_set)
    model.train()

    for batch in range(size):
        X = token_to_one_hot_tensor(X_set[batch])
        y = token_to_one_hot_tensor(y_set[batch])
        X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 10 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Test
def test(data, model, loss_fn):
    X_set, y_set = prepare_dataset(data)
    size = len(X_set)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in range(size):
            X = token_to_one_hot_tensor(X_set[batch])
            y = token_to_one_hot_tensor(y_set[batch])
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)

            # Compute prediction error
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax() == y.argmax()).type(torch.float).item()

    test_loss /= size
    correct /= size

    # Maximum accuracy possible on the test set: (33% + 100% + 50% + 100% + 100%) / 5 = ~78%
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Loss function, optimizer and training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

should_train = load_model_path == None
if load_model_path:
    model.load_state_dict(torch.load(load_model_path, weights_only=True))
else:
    epochs = 10000
    for t in range(epochs):
        train(input_data, model, loss_fn, optimizer)
        if t % 100 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            test(input_data, model, loss_fn)

        if save_model_path:
            torch.save(model.state_dict(), save_model_path)

    

# Inference
class InferenceModel:
    def __init__(self, model: NeuralNetwork, use_argmax: bool):
        self.model = model
        model.eval()
        
        # Use argmax on true, multinomial (random sample) on false
        self.use_argmax = use_argmax
    
    def __call__(self, inp):
        # Process input (word to one-hot)
        X = token_to_one_hot_tensor(inp)

        # Predict
        pred = self.model(X)

        # Process output (probabilities to word)
        if self.use_argmax:
            out_index = torch.argmax(pred)
        else:
            out_index = torch.multinomial(pred, num_samples=1)
        
        result = tokens[out_index.item()]

        # print(result, pred.detach().numpy())

        return result


print("\n--------- Inference start ---------")
model.reset()
inf_model = InferenceModel(model, False)
story = ['Jane']
for i in range(100):
    story.append(inf_model(story[-1]))

for i in range(len(story)):
    print(story[i], end='')
    if i < len(story)-1 and story[i] != '\n' and story[i+1] != '.':
        print(" ", end='')

print("\n---------- Inference end ----------")