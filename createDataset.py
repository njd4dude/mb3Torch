import json

# Define the range of x values
x_values = range(1, 50000 + 1)  # Example: 0 to 10

# Generate the dataset
dataset = [{'X': x, 'Y': 3 * x} for x in x_values]

# Specify the filename
filename = 'train_dataset.json'

# Write dataset to JSON file
with open(filename, 'w') as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset has been written to {filename}")
