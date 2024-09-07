# This testPredict.py takes the prediction model created in mb3.py and loads it to make predictions
import torch
from torch import nn
import pickle

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# Define model/class definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# Load model
model = torch.load("mb3Model.pth")

# load previous standardization parameters
with open("mb3MeanStd.pkl", "rb") as f:
    input_mean, input_std, target_mean, target_std = pickle.load(f)


print(input_mean, input_std, target_mean, target_std)

# prepare the for prediction by getting the input and standardizing it
model.eval()
input = torch.tensor(
    [1, 2, 10, 100, 1000, 10000, 1000000, 10000000], dtype=torch.float32
).reshape(-1, 1)
# standardize the input
input = (input - input_mean) / input_std

with torch.no_grad():
    # predict
    model.eval()
    input = input.to(device)
    pred = model(input)
    torch.set_printoptions(precision=8, sci_mode=False)

    # unstandardize the input
    input = (input * input_std) + input_mean
    print("input: ", input)

    # unstandardize the prediction
    pred = (pred * target_std) + target_mean
    print("pred: ", pred)
    print("\n")

    # print model parameters( to see weights and biases )
    for name, param in model.named_parameters():
        print(f"{name}: {param}")
