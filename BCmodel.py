import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import sklearn
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from timeit import default_timer as timer

!pip install torchmetrics # google colab
from torchmetrics import Accuracy

# Hyper parameters
N_SAMPLES = 2500
N_FEATURES = 2
N_CLASSES = 2
RANDOM_SEED = 235
BATCHES = 50
SHUFFLE = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

### ----
# Prepare data/model

# Features / Labels
X, y = make_gaussian_quantiles(n_samples=N_SAMPLES,
                               n_features=N_FEATURES,
                               n_classes=N_CLASSES,
                               random_state=RANDOM_SEED)

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# Setup train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=RANDOM_SEED,
                                                    shuffle=SHUFFLE)

# Setup train/test datasets/loaders
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset,
                          batch_size=BATCHES,
                          shuffle=SHUFFLE,)

test_dataset = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(dataset=test_dataset,
                         batch_size=BATCHES)

# Create model class
class GaussianQuantileModel_1(nn.Module):
  def __init__(self, input_features, output_features, neurons):
    super().__init__()

    self.layer_stack = nn.Sequential(
         nn.Linear(in_features=input_features,
                   out_features=neurons),
         nn.ReLU(),
         nn.Linear(in_features=neurons,
                   out_features=neurons),
         nn.ReLU(),
         nn.Linear(in_features=neurons,
                   out_features=output_features)
         )

  def forward(self, x):
    return self.layer_stack(x)

model_1 = GaussianQuantileModel_1(input_features=2,
                                  output_features=1,
                                  neurons=8).to(DEVICE)

# Setup loss, optimizer, and metrics
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(),
                             lr=0.03)
acc_fn = Accuracy(task="binary",
                  num_classes=N_CLASSES).to(DEVICE)

# Get untrained sample predictions
for X_batch, y_batch in test_dataloader:
  X_batch, y_batch = X_batch.to(DEVICE)[:10], y_batch.to(DEVICE)[:10]

  sample_logits = model_1(X_batch).squeeze()
  print(f"Sample logits: {sample_logits}")

  sample_pred_probs = torch.sigmoid(sample_logits)
  print(f"Sample pred probs: {sample_pred_probs}")

  sample_pred_labels = torch.round(sample_pred_probs)
  print(f"Sample pred labels:\t{sample_pred_labels}")
  print(f"Test pred labels:\t{y_batch}")

  sample_acc = acc_fn(preds=sample_pred_labels,
               target=y_batch)
  print(f"Sample accuracy: {sample_acc*100:.2f}%")

  break # Only want to predict on a single sample
  

### ----
# Train / Test loop 
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

EPOCHS = 1000

batches_processed = 0

time_start_gpu = timer()
for epoch in tqdm(range(EPOCHS)):
  train_loss, train_acc = 0, 0
  for X_train, y_train in train_dataloader:
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)

    model_1.train()
    y_logits = model_1(X_train).squeeze()
    y_pred_probs = torch.sigmoid(y_logits)
    y_pred_labels = torch.round(y_pred_probs)

    loss = loss_fn(y_logits, y_train)
    train_loss += loss
    acc = acc_fn(y_pred_labels, y_train)
    train_acc += acc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batches_processed += 1

  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  test_loss, test_acc = 0, 0
  model_1.eval()
  with torch.inference_mode():
    for X_test, y_test in test_dataloader:
      X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

      test_logits = model_1(X_test).squeeze()
      test_pred_probs = torch.sigmoid(test_logits)
      test_pred_labels = torch.round(test_pred_probs)

      test_loss += loss_fn(test_logits, y_test)
      test_acc += acc_fn(test_pred_labels, y_test)

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

  if epoch % 50 == 0:
    print(f"\n Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}%")
    print(f"\tBatches looked at: {batches_processed}")

time_end_gpu = timer()

time_total_gpu = time_end_gpu - time_start_gpu
print(f"Train time on GPU: {time_total_gpu:.3f} seconds | {time_total_gpu/60:.2f} minutes.")
