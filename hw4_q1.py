################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

from matplotlib import pyplot as plt

from MyPCA import MyPCA

from hw4_utils import load_MNIST, convert_data_to_numpy, plot_points

np.random.seed(2023)

normalize_vals = (0.1307, 0.3081)

batch_size = 100

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

# convert to numpy
X, y = convert_data_to_numpy(train_dataset)

#####################
# ADD YOUR CODE BELOW
#####################

model = MyPCA(2)
model.fit(X)
x_trans = model.project(X)

#plot
cmap = plt.get_cmap('Set1')

fig, ax = plt.subplots()

scatter = ax.scatter(x_trans[:,0], x_trans[:,1], c=y, cmap=cmap)
legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Digits")
ax.add_artist(legend1)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('MNIST Transformed')
plt.show()