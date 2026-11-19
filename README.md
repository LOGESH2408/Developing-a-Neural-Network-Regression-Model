# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1126" height="856" alt="image" src="https://github.com/user-attachments/assets/36d601be-559f-4f47-9f04-ce8dd8dc2309" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Logeshwaran S 

### Register Number:212225220058

```python
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/drive/MyDrive/DL EXP-1 - Sheet1.csv')
X = dataset1[['input']].values
y = dataset1[['output']].values

dataset1.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name: MOUIDHARAN.S   
# Register Number: 212224240095
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)

#Name: MOULIDHARAN.S
#Register Number: 212224240095
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
# train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
# This cell was commented out to avoid the NameError until the function is defined.

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')


```

### Dataset Information
<img width="262" height="415" alt="585528916-ab6ec503-f9d5-4931-bbb6-c27a43dc9520" src="https://github.com/user-attachments/assets/42fd19e5-07ee-48a8-af84-b6500fb2b8aa" />

### OUTPUT
<img width="202" height="245" alt="image" src="https://github.com/user-attachments/assets/9effcdf0-df63-4b12-93aa-6491d3083448" />
<img width="334" height="192" alt="image" src="https://github.com/user-attachments/assets/fe544451-2aa5-4285-878f-2f66f1b2f6d5" />
<img width="263" height="36" alt="image" src="https://github.com/user-attachments/assets/e8188985-d8b9-4031-a43c-ee4942a8d6bb" />


### Training Loss Vs Iteration Plot
<img width="689" height="469" alt="image" src="https://github.com/user-attachments/assets/d6ea755a-d2d2-4310-a7eb-aa66510b7704" />


### New Sample Data Prediction

<img width="344" height="33" alt="image" src="https://github.com/user-attachments/assets/16910d16-5ae1-487d-aa43-62f45caa9e8a" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
