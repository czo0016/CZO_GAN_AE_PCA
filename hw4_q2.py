################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data_utils

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from MyAutoencoder import MyAutoencoder

from hw4_utils import load_MNIST, plot_points

np.random.seed(2023)

batch_size = 10

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################
#

model = MyAutoencoder(28*28,400,2,learning_rate=.001,max_epochs=2)
loss_list = model.fit(train_loader)

train_subset = data_utils.Subset(train_dataset, range(1000))
encoded_samples, labels = model.encode(train_subset)

digits =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cmap = plt.get_cmap('Set1')
num_colors = len(digits)
color_map = {i: cmap(i/num_colors) for i in range(num_colors)}

#plot
fig,ax = plt.subplots()

for data,label in zip(encoded_samples, labels):
    color = color_map[label]
    ax.plot(data[0][0], data[0][1], color=color, marker='o',markersize=5, label=digits[label])

#legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1,0.5))

ax.set_xlabel('EHN1')
ax.set_ylabel('EHN2')
plt.show()

x_values = range(1,len(loss_list)+1)
plt.plot(x_values, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()