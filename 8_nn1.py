import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


is_cuda_available = torch.cuda.is_available()

dataset = np.loadtxt(os.path.join("data", "nn1_data.csv"), delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


# define the model
class BinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x


model = BinClassifier()
if is_cuda_available:
    os.environ["ROCBLAS_LAYER"] = "1"
    X = X.to("cuda")
    y = y.to("cuda")
    model = model.to("cuda")
print(model)

# train the model
loss_fn = nn.BCELoss()  # binary cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy*100}%")

predictions = (model(X) > 0.5).int()
for i in range(5):
    print(f'{X[i].tolist()} => {predictions[i]} (expected {y[i]})')
