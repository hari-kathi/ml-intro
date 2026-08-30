# PyTorch cheat sheet

*One page for mid-exercise lookups. Taught in Modules 09–11 — see the [curriculum](../curriculum.md#module-10--pytorch-fundamentals), and the [glossary](../glossary.md) for terms like tensor, autograd, epoch, batch.*

## Tensors — NumPy arrays that remember their math

```python
import torch
import numpy as np

torch.manual_seed(42)                    # reproducible randomness, like np.random.default_rng(42)

x = torch.tensor([1.0, 2.0, 3.0])        # make a tensor from a Python list
z = torch.zeros(3, 4)                    # 3x4 tensor of zeros (ones() and rand() work the same way)
x.shape                                  # dimensions, just like NumPy's .shape
x * 2 + 1                                # element-wise math, just like NumPy

t = torch.from_numpy(np_array).float()   # NumPy -> tensor (.float() makes it 32-bit, the standard)
back = t.numpy()                         # tensor -> NumPy (for plotting with matplotlib)

# autograd: the superpower NumPy doesn't have
w = torch.tensor(2.0, requires_grad=True)  # requires_grad=True: "record what happens to me"
loss = (w * 3 - 6) ** 2                    # any calculation built from w
loss.backward()                            # compute d(loss)/dw automatically
w.grad                                     # ...and here it is
```

## nn.Module skeleton — how every model is written

```python
import torch.nn as nn                    # nn = the neural-network building blocks

class Net(nn.Module):                    # every model is a class inheriting from nn.Module
    def __init__(self):
        super().__init__()               # run nn.Module's own setup first (required line)
        # define the layers as attributes; sizes must chain: 13 -> 16 -> 1
        self.hidden = nn.Linear(13, 16)  # 13 input features -> 16 hidden neurons
        self.act = nn.ReLU()             # activation: the nonlinearity between layers
        self.out = nn.Linear(16, 1)      # 16 hidden -> 1 output score

    def forward(self, x):                # forward() defines how data flows through
        x = self.hidden(x)               # weighted sums, layer 1
        x = self.act(x)                  # nonlinearity
        x = self.out(x)                  # final score (a "logit", pre-probability)
        return x

model = Net()                            # build it
model(some_input)                        # calling the model runs forward()
```

## The canonical training loop — the liturgy that never changes

```python
loss_fn = nn.BCEWithLogitsLoss()         # loss for yes/no classification
                                         # (nn.CrossEntropyLoss for multi-class,
                                         #  nn.MSELoss for regression)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                                         # the optimizer updates the weights;
                                         # lr = learning rate (step size downhill)

for epoch in range(100):                 # one epoch = one full pass over the data
    predictions = model(X_train)         # 1. FORWARD: run data through the model
    loss = loss_fn(predictions, y_train) # 2. LOSS: one number for "how wrong?"
    optimizer.zero_grad()                # 3. clear old gradients (they ACCUMULATE
                                         #    by default -- forgetting this is the
                                         #    classic PyTorch bug)
    loss.backward()                      # 4. BACKWARD: autograd computes the
                                         #    gradient of loss w.r.t. every weight
    optimizer.step()                     # 5. STEP: nudge every weight downhill
    if epoch % 10 == 0:                  # % is remainder: true every 10th epoch
        print(epoch, loss.item())        # .item() turns a 1-value tensor into a
                                         #    plain Python number for printing
```

Forward → loss → zero_grad → backward → step. This exact five-line cycle trains everything from Module 09's two-moons toy to GPT.

## Evaluating (gradients off)

```python
model.eval()                             # evaluation mode (matters for dropout etc.)
with torch.no_grad():                    # no_grad: don't record -- faster, less memory
    test_predictions = model(X_test)
model.train()                            # back to training mode before more training
```

## Batches with DataLoader

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_train, y_train)          # zips features and labels together
loader = DataLoader(dataset, batch_size=32, shuffle=True)  # deals out shuffled batches of 32

for epoch in range(20):
    for X_batch, y_batch in loader:      # inner loop: one gradient step per batch
        predictions = model(X_batch)     # ...then the same five-step liturgy as above
        loss = loss_fn(predictions, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Device handling (CPU vs Apple GPU)

```python
# pick the fastest available device; "mps" is the Apple-silicon GPU
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = model.to(device)                 # move the model's weights to that device
X_batch = X_batch.to(device)             # data must be moved too --
y_batch = y_batch.to(device)             # model and data must be on the SAME device

result = predictions.cpu().numpy()       # bring outputs back to the CPU before NumPy/matplotlib
```

Lessons default to CPU for reproducibility; MPS is worth it for the image models in Module 11 and the capstone.

## Saving and loading

```python
torch.save(model.state_dict(), "model.pt")   # save just the learned weights
model = Net()                                # rebuild the architecture first...
model.load_state_dict(torch.load("model.pt", weights_only=True))  # ...then pour the weights back in
```
