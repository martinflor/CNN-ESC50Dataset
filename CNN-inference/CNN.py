# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 22:40:40 2022

@author: Ineed
"""


import torch
import torch.nn as nn
import numpy as np

def training(model, train_dl, num_epochs):
  
  acc_train = []
  acc_test = []
  
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        #if i % 10 == 0:    # print every 10 mini-batches
        #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    acc_train.append(acc)
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Train Accuracy: {acc:.2f}')
    
    acc_test.append(inference(model, val_dl))
    

  print('Finished Training')
  return acc_train, acc_test
  





def inference (model, val_dl):
  correct_prediction = 0
  total_prediction = 0

  # Disable gradient updates
  with torch.no_grad():
    for data in val_dl:
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
    
  acc = correct_prediction/total_prediction
  print(f'Test Accuracy: {acc:.2f}, Total items: {total_prediction}')
  
  return acc







from MelSpecDS import MSDS

myds = MSDS("datas/dataset.csv")


from torch.utils.data import random_split

num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])


train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)



from Network import MelSpecClassifier


myModel = MelSpecClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
next(myModel.parameters()).device




import time

t1 = time.time()
num_epochs= 20
trainAcc, testAcc = training(myModel, train_dl, num_epochs)

#inference(myModel, val_dl)

t2= time.time()

print("Time needed : ", t2-t1)

import matplotlib.pyplot as plt

x = np.arange(0, num_epochs, 1)

plt.plot(x, trainAcc, label = 'train')
plt.plot(x, testAcc, label = 'test')
plt.grid()
plt.legend()
plt.xlabel("Number of Epochs")
plt.ylabel("Train/Test accuracy")
plt.title("Train/Test accuracy with regards to the number of epochs")
plt.show()

