import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
import numpy as np
import pandas as pd

#prepare data
data = np.load("mnist_kaggle.npz")

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

x_train = torch.tensor(x_train, dtype=torch.float32) / 255.0
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32) / 255.0

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)

#model 
#construct model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(784,512)
        self.linear2 = torch.nn.Linear(512,256)
        self.linear3 = torch.nn.Linear(256,128)
        self.linear4 = torch.nn.Linear(128,64)
        self.linear5 = torch.nn.Linear(64,10)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        y_pred = self.linear5(x)
        return y_pred

model = Net()

#Loss and optim
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#model training
def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader , 0):
        x , y = data

        y_pred = model(x)
        loss = criterion(y_pred , y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)



#prepare test data and create output
test_dataset = TensorDataset(x_test)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False)

model.eval()

predictions = []

with torch.no_grad():
    for (inputs,) in test_loader:

        outputs = model(inputs)

        predicted = torch.argmax(outputs, dim=1)

        predictions.extend(predicted.cpu().numpy())


submission = pd.DataFrame({
    "Id": range(0, len(predictions)),
    "Category": predictions
})

submission.to_csv("submission.csv", index=False)


