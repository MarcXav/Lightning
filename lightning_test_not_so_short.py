import os
import sys
sys.path.insert(0, '..')
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

from python_environment_check import check_packages


d = {
    'torch': '1.8',
    'torchvision': '0.9.0',
    'tensorboard': '2.7.0',
    'pytorch_lightning': '1.5.0',
    'torchmetrics': '0.6.2'
}
check_packages(d)

"""
Define a LightningModule

A LightningModule enables your PyTorch nn.Module to play together in complex 
ways inside the training_step (there is also an optional validation_step and 
test_step).
"""

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)

"""
Define a dataset

Lightning supports ANY iterable (DataLoader, numpy, etc…) for the 
train/val/test/predict splits.
"""

# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

"""
Train the model

The Lightning Trainer “mixes” any LightningModule with any dataset and abstracts 
away all the engineering complexity needed for scale.

See documentation in lightning.ai
"""

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = Trainer()

stop 

trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

"""
Use the model

Once you’ve trained the model you can export to onnx, torchscript and put it 
into production or simply load the weights and run predictions.
"""

# load checkpoint
checkpoint = "./lightning_logs/version_48/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# embed 4 fake images!
fake_image_batch = Tensor(4, 28 * 28)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

"""
Visualize training

If you have tensorboard installed, you can use it for visualizing experiments.

Run this on your commandline and open your browser to http://localhost:6006/
tensorboard --logdir .

"""

"""
Supercharge training

Enable advanced training features using Trainer arguments. These are 
state-of-the-art techniques that are automatically integrated into your training 
loop without changes to your code.

# train on 4 GPUs
trainer = Trainer(
    devices=4,
    accelerator="gpu",
 )

train 1TB+ parameter models with Deepspeed/fsdp
trainer = Trainer(
    devices=4,
    accelerator="gpu",
    strategy="deepspeed_stage_2",
    precision=16
 )

# 20+ helpful flags for rapid idea iteration
trainer = Trainer(
    max_epochs=10,
    min_epochs=5,
    overfit_batches=1
 )

# access the latest state of the art techniques
trainer = Trainer(callbacks=[StochasticWeightAveraging(...)])
"""

import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision import datasets


"""
Add a test loop

To make sure a model can generalize to an unseen dataset (ie: to publish a paper 
or in a production environment) a dataset is normally split into two parts, the 
train split and the test split.

The test set is NOT used during training, it is ONLY used once the model has 
been trained to see how the model will do in the real-world.

Find the train and test splits

Datasets come with two splits. Refer to the dataset documentation to find the 
train and test splits.
"""

# define any number of nn.Modules (or use your current ones)

# Load data sets
transform = transforms.ToTensor()
train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

"""
Define the test loop

To add a test loop, implement the test_step method of the LightningModule
"""
"""
Train with the test loop

Once the model has finished training, call .test
"""

from torch.utils.data import DataLoader

# initialize the Trainer
trainer = Trainer()

# test the model
trainer.test(model, dataloaders=DataLoader(test_set))

"""
Add a validation loop

During training, it’s common practice to use a small portion of the train split 
to determine when the model has finished training.
As a rule of thumb, we use 20% of the training set as the validation set. 
This number varies from dataset to dataset.
"""

# Split the training data
# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

