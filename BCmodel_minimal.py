import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
 
import sklearn
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
 
### ----
# Prepare data/model
 
# Features / Labels
X, y = make_gaussian_quantiles(n_samples=2500,
                               n_features=2,
                               n_classes=2,
                               random_state=235)
 
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)
 
# Setup train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=235,
                                                    shuffle=True)
 
# Setup train/test datasets/loaders
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset,
                          batch_size=50,
                          shuffle=True)
 
test_dataset = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(dataset=test_dataset,
                         batch_size=50)
 
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
                                  neurons=8)
 
# Setup loss, optimizer, and metrics
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(),
                             lr=0.03)
 
### ----
# Train / Test loop 
torch.manual_seed(235)
torch.cuda.manual_seed(235)

for epoch in range(1000):
  train_loss, train_acc = 0, 0
  for X_train, y_train in train_dataloader: 
    model_1.train()
    y_logits = model_1(X_train).squeeze()
    y_pred_probs = torch.sigmoid(y_logits)
    y_pred_labels = torch.round(y_pred_probs)
 
    loss = loss_fn(y_logits, y_train)
    train_loss += loss
    acc = (y_pred_labels == y_train).sum().item() / y_train.size(0)
    train_acc += acc
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)
 
  test_loss, test_acc = 0, 0
  model_1.eval()
  with torch.inference_mode():
    for X_test, y_test in test_dataloader: 
      test_logits = model_1(X_test).squeeze()
      test_pred_probs = torch.sigmoid(test_logits)
      test_pred_labels = torch.round(test_pred_probs)
 
      test_loss += loss_fn(test_logits, y_test)
      test_acc += (test_pred_labels == y_test).sum().item() / y_test.size(0)
 
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
 
  if epoch == 1000:
    print(f"Train loss: {train_loss:.3f}, Train acc: {train_acc*100:.2f}% | Test loss: {test_loss:.3f}, Test acc: {test_acc*100:.2f}%")
