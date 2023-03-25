# coding: utf-8


import sys
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import sklearn
import sklearn.model_selection
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
import torchvision 
from torchvision import transforms 

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples


# # Chapter 13: Going Deeper -- the Mechanics of PyTorch (Part 2/3)

# **Outline**
# 
# - [Project one - predicting the fuel efficiency of a car](#Project-one----predicting-the-fuel-efficiency-of-a-car)
#   - [Working with feature columns](#Working-with-feature-columns)
#   - [Training a DNN regression model](#Training-a-DNN-regression-model)
# - [Project two - classifying MNIST handwritten digits](#Project-two----classifying-MNIST-handwritten-digits)

# ## Project one - predicting the fuel efficiency of a car
# 

# ### Working with feature columns
# 
# 

# Setting up dataframe, pre-processing using PyTorch ***************************

print(f"Working with feature columns", end='\n\n')
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

df = pd.read_csv(url, names=column_names,
                 na_values = "?", comment='\t',
                 sep=" ", skipinitialspace=True)

print(f"MpG dataframe, shape: {df.shape}")
print(df.tail())
print()

print(f"Attributes with null values")
print(df.isna().sum())
print(f"\tdropping rows with nulls...",end="\n\n")
df = df.dropna()
df = df.reset_index(drop=True)

print(f"MpG shortened dataframe, shape: {df.shape}")
print(df.tail())
print()

df_train, df_test = sklearn.model_selection.train_test_split(df, train_size=0.8, random_state=1)
train_stats = df_train.describe().transpose()
print("Dataset statistics")
print(train_stats)
print()

numeric_column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']

df_train_norm, df_test_norm = df_train.copy(), df_test.copy()

for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean']
    std  = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean)/std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean)/std

print(f"MpG dataframe with normalized numeric columns")   
print(df_train_norm.tail())
print()


# Setting up year buckets (0 for year <73, 1 for year >=73 and <76 etc...)
print(f"Setting up year buckets")
# Set boundary list
boundaries = torch.tensor([73, 76, 79])
 
# Set up buckets for training dataset
v = torch.tensor(df_train_norm['Model Year'].values)
df_train_norm['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)

# Set up buckets for test dataset
v = torch.tensor(df_test_norm['Model Year'].values)
df_test_norm['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)

# Add attribute to the list of numeric columns
numeric_column_names.append('Model Year Bucketed')


# Define list for the unordered categorical feature (Origin)
print(f"Define list for the unordered categorical feature",end="\n\n")
total_origin = len(set(df_train_norm['Origin']))

# Apply one-hot encoding to training set
origin_encoded = one_hot(torch.from_numpy(df_train_norm['Origin'].values) % total_origin)
# Concatenate origine encoded with other numeric data
x_train_numeric = torch.tensor(df_train_norm[numeric_column_names].values)
x_train = torch.cat([x_train_numeric, origin_encoded], 1).float()
 
# Apply one-hot encoding to training set
origin_encoded = one_hot(torch.from_numpy(df_test_norm['Origin'].values) % total_origin)
# Concatenate origine encoded with other numeric data
x_test_numeric = torch.tensor(df_test_norm[numeric_column_names].values)
x_test = torch.cat([x_test_numeric, origin_encoded], 1).float()

# Create label tensors
y_train = torch.tensor(df_train_norm['MPG'].values).float()
y_test = torch.tensor(df_test_norm['MPG'].values).float()


# Training a DNN regression model **********************************************
print(f"1. Training a DNN regression model", end="\n\n")

# Creating the dataloader
print(f"Create dataloader")
train_ds = TensorDataset(x_train, y_train)
batch_size = 8
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Building the model with 2 fully connected layers where one has 8 hidden units
# and the other has 4
print(f"Build model")
hidden_units = [8, 4]
input_size = x_train.shape[1]

all_layers = []
for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit

all_layers.append(nn.Linear(hidden_units[-1], 1))

model = nn.Sequential(*all_layers)

print(f"DNN regression model for MpG")
print(model)
print()

# Define loss function and optimizer
print(f"Defined loss function and optimizer")
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train the model for 200 epochs
print(f"Training the model for 200 epochs:")
torch.manual_seed(1)
num_epochs = 200
log_epochs = 20 
for epoch in range(num_epochs):
    loss_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)[:, 0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train += loss.item()
    if epoch % log_epochs==0:
        print(f'Epoch {epoch}  Loss {loss_hist_train/len(train_dl):.4f}')

with torch.no_grad():
    pred = model(x_test.float())[:, 0]
    loss = loss_fn(pred, y_test)
    print(f'Test MSE: {loss.item():.4f}')
    print(f'Test MAE: {nn.L1Loss()(pred, y_test).item():.4f}')
print()


# ## Project two - classifying MNIST hand-written digits ***********************

print(f"2. Classifying MNIST hand-written digits", end='\n\n')
print(f"Load data")
image_path = './'

# Apply a custome transformation to convert pixels to floating type tensor 
# and normalize the data, using the 'ToTensor()' method
transform = transforms.Compose([transforms.ToTensor()])

mnist_train_dataset = torchvision.datasets.MNIST(root=image_path, 
                                           train=True, 
                                           transform=transform, 
                                           download=True)
mnist_test_dataset = torchvision.datasets.MNIST(root=image_path, 
                                           train=False, 
                                           transform=transform, 
                                           download=False)
# Creating the dataloader
print(f"Create dataloader") 
batch_size = 64
torch.manual_seed(1)
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)

# Building the model with 1 fully connected layers where one has 32 hidden units
# and the other has 16
print(f"Build model")
hidden_units = [32, 16]
image_size = mnist_train_dataset[0][0].shape
input_size = image_size[0] * image_size[1] * image_size[2]

all_layers = [nn.Flatten()]
for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit

all_layers.append(nn.Linear(hidden_units[-1], 10))
model = nn.Sequential(*all_layers)

print(f"Model for classifying MNIST hand-written digits")
print(model)
print()

# Define loss function and optimizer
print(f"Defined loss function and optimizer")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model for 200 epochs
print(f"Training the model for 20 epochs:")
torch.manual_seed(1)
num_epochs = 20
for epoch in range(num_epochs):
    accuracy_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist_train += is_correct.sum()
    accuracy_hist_train /= len(train_dl.dataset)
    print(f'Epoch {epoch}  Accuracy {accuracy_hist_train:.4f}')
print()

pred = model(mnist_test_dataset.data / 255.)
is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
print(f'Test accuracy: {is_correct.mean():.4f}')  


# ---
