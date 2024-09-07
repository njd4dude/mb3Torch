import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
# Note: Here I used the wrong term "validation" instead of "validation" so it should be renamed.
# define constants
learning_rate = 1
batch_size = 50
epochs = 5


def parse_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    # extract inputX and targetY
    inputX = [object["X"] for object in data]
    targetY = [object["Y"] for object in data]

    # convert list to tensor
    input_tensor = torch.tensor(inputX, dtype=torch.float32)
    target_tensor = torch.tensor(targetY, dtype=torch.float32)

    return (input_tensor, target_tensor)


def get_training_data_mean_std(input_tensor, target_tensor):
    input_mean = input_tensor.mean()
    input_std = input_tensor.std()

    target_mean = target_tensor.mean()
    target_std = target_tensor.std()

    return (input_mean, input_std, target_mean, target_std)


def standardize_data(tensor, mean, std):
    standardized_data = (tensor - mean) / std
    return standardized_data


# create custom dataset class
class CustomImageDataset(Dataset):
    def __init__(
        self, input_tensor, target_tensor, transform=None, target_transform=None
    ):
        self.features = input_tensor
        self.labels = target_tensor
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = feature.unsqueeze(0)
        label = label.unsqueeze(0)

        # Apply transformations if specified
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)

        return feature, label


# parse the json file for the tensors
train_input_tensor, train_target_tensor = parse_json_file("train_dataset.json")
validation_input_tensor, validation_target_tensor = parse_json_file("validation_dataset.json")

# get the mean and std of the input and target tensors from ONLY the training data. This important that we use the training data only to
# calculate the mean and std so that the scale of the standardized data is consistent
input_mean, input_std, target_mean, target_std = get_training_data_mean_std(
    train_input_tensor, train_target_tensor
)

# dump the mean and std values to a pickle file
with open("mb3MeanStd.pkl", "wb") as file:
    pickle.dump((input_mean, input_std, target_mean, target_std), file)

# standardized TRAINING data
standardized_training_input_tensor = standardize_data(
    train_input_tensor, input_mean, input_std
)
standardized_training_target_tensor = standardize_data(
    train_target_tensor, target_mean, target_std
)
# standardized validation data
standardized_validation_input_tensor = standardize_data(
    validation_input_tensor, input_mean, input_std
)
standardized_validation_target_tensor = standardize_data(
    validation_target_tensor, target_mean, target_std
)


# create custom dataset instances
train_mb3Dataset = CustomImageDataset(
    standardized_training_input_tensor, standardized_training_input_tensor
)

validation_mb3Dataset = CustomImageDataset(
    standardized_validation_input_tensor, standardized_validation_target_tensor
)


# create data loaders
train_mb3_dataloader = DataLoader(train_mb3Dataset, batch_size=batch_size, shuffle=True)
validation_mb3_dataloader = DataLoader(validation_mb3Dataset, batch_size=batch_size, shuffle=True)


def print_mb3_dataloader():
    i = 0
    for x, y in validation_mb3_dataloader:
        print("batch i: ", i)
        print("x: ", x)
        print("y: ", y)
        print("x shape: ", x.shape)
        print("y shape: ", y.shape)
        i += 1
        break


print_mb3_dataloader()


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(1, 1))

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# Create an instance of the model
model = NeuralNetwork().to(device)
print(model)

# Define loss function and optimizer
loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# define training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print("num_batches for train: ", num_batches)
    model.train()
    epoch_train_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)
        epoch_train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            # print(f"loss: {loss:>10f}  [{current:>5d}/{size:>5d}]")
            print(f"loss: {loss}  [{current:>5d}/{size:>5d}]") #changed to this so i can see the raw loss values



    epoch_train_loss /= num_batches
    return epoch_train_loss


# define validation function
def validation(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    print("num_batches for validation: ", num_batches)

    model.eval()
    validation_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            validation_loss += loss_fn(pred, y).item()
    validation_loss /= num_batches
    print(f"validation Error: \n Avg loss (MSE): {validation_loss:>8f} \n")
    return validation_loss


train_losses = []
validation_losses = []

# Start the training
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    validation_loss = validation(validation_mb3_dataloader, model, loss_fn)
    train_loss = train(train_mb3_dataloader, model, loss_fn, optimizer)
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)

# plot the train losses
plt.plot(
    range(1, len(train_losses) + 1),
    train_losses,
    label="Train Loss (MSE)",
    color="red",
)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.title("Train Loss Over Epochs")
plt.legend()
plt.show()

# plot the validation losses
plt.plot(
    range(1, len(validation_losses) + 1),
    validation_losses,
    label="validation Loss (MSE)",
)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.title("validation Loss Over Epochs")
plt.legend()
plt.show()


# Save the model weights
torch.save(model.state_dict(), "mb3Weights.pth")
print("Saved PyTorch Model State to mb3Weights.pth")

# save the whole model
torch.save(model, "mb3Model.pth")
print("Saved PyTorch Model to mb3Model.pth")

# make predictions
model.eval()
input = torch.tensor([2], dtype=torch.float32)
# standardize the input
print("regular input: ", input)
input = (input - input_mean) / input_std
print("standardized input: ", input)
with torch.no_grad():
    input = input.to(device)
    pred = model(input)

    pred = (pred * target_std) + target_mean

    print("regular pred: ", pred)

