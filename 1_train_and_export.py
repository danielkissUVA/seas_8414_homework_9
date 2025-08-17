# Filename: 1_train_and_export.py
# This script creates a sample dataset, trains an H2O AutoML model
# to classify domains, and then exports the best model.

import csv
import random
import math
import h2o
from h2o.automl import H2OAutoML
import os

def get_entropy(s):
    """
    Calculates the Shannon entropy of a string.
    Entropy is a measure of randomness. Higher entropy suggests a more random string.
    """
    p, lns = {}, float(len(s))
    for c in s:
        p[c] = p.get(c, 0) + 1
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

# --- Step 1: Create Sample Dataset ---
print("Creating sample dataset 'dga_dataset_train.csv'...")

header = ['domain', 'length', 'entropy', 'class']
data = []
# Create a list of sample legitimate domains.
legit_domains = ['google', 'facebook', 'amazon', 'github', 'wikipedia', 'microsoft']
for _ in range(100):
    domain = random.choice(legit_domains) + ".com"
    data.append([domain, len(domain), get_entropy(domain), 'legit'])

# Create a list of sample DGA domains (random strings of a certain length).
for _ in range(100):
    length = random.randint(15, 25)
    domain = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(length)) + ".com"
    data.append([domain, len(domain), get_entropy(domain), 'dga'])

# Write the data to a CSV file.
with open('dga_dataset_train.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print("dga_dataset_train.csv created successfully.")

# --- Step 2: Initialize H2O and Train AutoML Model ---
print("\nInitializing H2O...")
h2o.init()

print("Importing training data...")
train = h2o.import_file("dga_dataset_train.csv")

# Define features and the target variable.
x = ['length', 'entropy'] # Features
y = "class"               # Target

# Convert the target variable to a factor (categorical) type, as required by H2O.
train[y] = train[y].asfactor()

print("Starting H2O AutoML process...")
# Initialize AutoML with parameters to control the training process.
# max_models: maximum number of models to train.
# max_runtime_secs: maximum time to spend training models.
aml = H2OAutoML(max_models=20, max_runtime_secs=120, seed=1)
# Train the AutoML model.
aml.train(x=x, y=y, training_frame=train)

print("H2O AutoML process complete.")
print("Leaderboard:")
print(aml.leaderboard.head())

# --- Step 3: Export the Best Model ---
print("\nExporting the best model...")
# Get the best performing model from the leaderboard.
best_model = aml.leader

# Define the path for the model artifacts.
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# Download the MOJO (Model Object, Optimized) artifact.
# MOJO files are production-ready and can be used for predictions
# without the need for the full H2O environment.
mojo_path = best_model.download_mojo(path=model_dir)

# Define the new desired path and filename for the MOJO file.
new_mojo_path = os.path.join(model_dir, "DGA_Leader.zip")

# Rename the MOJO file to the specified name.
if os.path.exists(mojo_path):
    os.rename(mojo_path, new_mojo_path)
    print(f"Production-ready MOJO model saved to: {new_mojo_path}")
else:
    print(f"Could not find MOJO file at {mojo_path} to rename.")

# Save the best model in its native H2O format as well.
model_path = h2o.save_model(model=aml.leader, path=model_dir, force=True)
print(f"Original saved H2O model path: {model_path}")

# Rename the folder to a friendly name for easier identification.
custom_name = "best_dga_model"
new_path = os.path.join(model_dir, custom_name)

# Check if the path exists before renaming to avoid errors.
if os.path.exists(model_path):
    os.rename(model_path, new_path)
    print(f"Renamed H2O model path: {new_path}")
else:
    print(f"Could not find original H2O model path at {model_path} to rename.")

# --- Step 4: Shut Down H2O ---
print("\nShutting down H2O...")
h2o.shutdown()