
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.nn.functional as F
import os
from torchvision.models import resnet18
from torchvision.models import resnet152
from torchvision.models import efficientnet_b0

class ADHD_Dataset():
    def __init__(self):

        # whole dataset
        df = pd.read_csv('dataset_2.csv')
        self.y = df['K2Q31A']
        self.X = df.drop(columns='K2Q31A')

        # chi square
        self.chi = pd.read_csv('chi.csv')
        self.chi = self.chi.loc[self.chi['Dr Sheikhy'] == 'Y']['feature name']

        # fisher's score
        self.fisher = pd.read_csv('fisher.csv')
        self.fisher = self.fisher.loc[self.fisher['Dr Sheikhy'] == 'Y']['Feature Name']

        # information gain
        self.inf = pd.read_csv('inf-gain.csv')
        self.inf = self.inf.loc[self.inf['Dr Sheikhy'] == 'Y']['Feature Name']

        # corelation
        self.cor = pd.read_csv('cor.csv')
        self.cor = self.cor.iloc[:,[0, 14]].where(self.cor.iloc[:,14] == 'Y').dropna().iloc[:, 0]

    def return_dataset(self)->pd.DataFrame:
        return self.X, self.y

    def return_chi(self)->pd.Series:
        return self.chi

    def return_fisher(self)->pd.Series:
        return self.fisher

    def return_inf(self)->pd.Series:
        return self.inf

    def return_cor(self)->pd.Series:
        return self.cor

    # intersections of 2 sets
    def return_intersection_chi_fisher(self) -> list:
        return list(set(self.chi.tolist()) & set(self.fisher.tolist()))

    def return_intersection_chi_inf(self) -> list:
        return list(set(self.chi.tolist()) & set(self.inf.tolist()))

    def return_intersection_chi_cor(self) -> list:
        return list(set(self.chi.tolist()) & set(self.cor.tolist()))

    def return_intersection_fisher_inf(self) -> list:
        return list(set(self.fisher.tolist()) & set(self.inf.tolist()))

    def return_intersection_fisher_cor(self) -> list:
        return list(set(self.fisher.tolist()) & set(self.cor.tolist()))

    def return_intersection_inf_cor(self) -> list:
        return list(set(self.inf.tolist()) & set(self.cor.tolist()))

    # intersections of 3 sets
    def return_intersection_chi_fisher_inf(self) -> list:
        return list(set(self.chi.tolist()) & set(self.fisher.tolist()) & set(self.inf.tolist()))

    def return_intersection_chi_fisher_cor(self) -> list:
        return list(set(self.chi.tolist()) & set(self.fisher.tolist()) & set(self.cor.tolist()))

    def return_intersection_chi_inf_cor(self) -> list:
        return list(set(self.chi.tolist()) & set(self.inf.tolist()) & set(self.cor.tolist()))

    def return_intersection_fisher_inf_cor(self) -> list:
        return list(set(self.fisher.tolist()) & set(self.inf.tolist()) & set(self.cor.tolist()))



# Set environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.backends.cudnn.enabled = True


# model = resnet18(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, 2)
# Dataset
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return torch.tensor(x).float(), torch.tensor(y).long()

# Data preprocessing
data = ADHD_Dataset()

df = pd.read_csv('dataset_2.csv')  # Replace 'your_dataset.csv' with the actual file path
# X = df.drop(["K2Q31A"], axis=1).values

cols = data.return_cor().tolist()
X3 = df[cols]
X3 = df.drop(["K2Q31A"], axis=1).values
y = df["K2Q31A"].values
X = X3
dataset = ADHD_Dataset()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
y_train[y_train == 1.0] = 0.
y_train[y_train == 2.0] = 1.

y_test[y_test == 1.0] = 0.
y_test[y_test == 2.0] = 1.


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# DataLoader
batch_size = 32
train_dataset = TabularDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
test_dataset = TabularDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

# Model, loss and optimizer
input_dim = X_train.shape[1]
# model = TabularModel(input_dim)
# from torchvision.models import resnet1d
model = resnet18(pretrained=False)
# model = efficientnet_b0(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.cpu() # Start on CPU

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
device = torch.device("cuda")
model.to(device=device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for inputs, labels in progress_bar:
        inputs = inputs.repeat(1, 3, 1, 1)
        inputs = inputs.permute((2, 1, 0, 3))
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        progress_bar.set_postfix({"Train Loss": train_loss / ((progress_bar.n + 1) * inputs.size(0))})

    train_loss /= len(train_loader.dataset)

    # Evaluation on the test set
    from sklearn.metrics import accuracy_score

# Evaluation on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    predictions = []
    actuals = []
    accuracies = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.repeat(1, 3, 1, 1)
            inputs = inputs.permute((2, 1, 0, 3))
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            actuals.extend(labels.tolist())

    test_loss /= len(test_loader.dataset)

    # Convert predictions and actuals to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate accuracy
    accuracy = accuracy_score(actuals, predictions)
    accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f}")
average_accuracy = np.mean(accuracies)
print(f"Average Accuracy: {average_accuracy:.4f}")
