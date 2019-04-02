#Modules Loaded
import os
import glob
import torch
from typing import TypeVar, NamedTuple, List, Optional, Tuple
import cv2
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.autograd import Variable

def get_train_data(file_dir):
    files = os.listdir(file_dir)
    train_states = [] 
    for i in range(len(files)):
        img = cv2.imread(os.path.join(file_dir,files[i]))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        new_gray = cv2.resize(gray,(28,28))  
        train_states.append(new_gray)
    train_states = np.expand_dims(train_states, axis=1) 
    return train_states

def get_label(loc,file_dir):
    files = os.listdir(file_dir)
    tree = ET.parse(loc)
    root = tree.getroot()
    labels = []
    for i in range(len(files)):
        for child in root:
            if files[i] == child.attrib['file'].split('/')[2]:
                labels.append(child.attrib['tag'])
    unique_labels = np.unique(labels)
    labels_dict = {}
    for i in range(len(unique_labels)):
        labels_dict[unique_labels[i]] = i
    new_labels =[]
    for i in range(len(labels)):
        new_labels.append(labels_dict[labels[i]])
    return new_labels


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size = 3,stride= 1,padding=1)
        self.conv2 = nn.Conv2d(50, 10, kernel_size =3,stride = 1,padding =1 )
        self.fc1 = nn.Linear(28*28*10, 500)
        self.fc2 = nn.Linear(500, 59)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 28*28*10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x    
	
#Parameters of Model
model = Net()
batch_size = 50
num_epochs = 100
learning_rate = 0.01
file_dir ='char/char'
states = get_train_data(file_dir)
loc = 'char/char.xml'
new_labels = get_label(loc,file_dir)
batch_no = len(states) // batch_size
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training Step
for epoch in range(num_epochs):
    if epoch % 5 == 0:
        print('Epoch {}'.format(epoch+1))
    x_train, y_train = shuffle(states, new_labels)
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_var = Variable(torch.FloatTensor(x_train[start:end]))
        y_var = Variable(torch.LongTensor(y_train[start:end]))
        optimizer.zero_grad()
        ypred_var = model(x_var)
        loss =criterion(ypred_var, y_var)
        loss.backward()
        optimizer.step()
		
torch.save(model.state_dict(),'model.pt')