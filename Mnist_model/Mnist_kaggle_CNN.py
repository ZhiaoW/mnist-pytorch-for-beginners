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

x_train = x_train.reshape(-1, 1, 28,28)
x_test = x_test.reshape(-1, 1, 28,28)

x_train = torch.tensor(x_train, dtype=torch.float32) / 255.0
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32) / 255.0

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)

#model 
#construct model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10, kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(10,20, kernel_size = 5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.linear = torch.nn.Linear(320,10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x



model = CNN()

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
    for epoch in range(13):
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


