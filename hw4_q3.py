################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.utils.data as data_utils
from matplotlib import pyplot as plt

from MyGenerator import MyGenerator
from MyDiscriminator import MyDiscriminator

from hw4_utils import load_MNIST

np.random.seed(2023)

batch_size = 128

normalize_vals = (0.5, 0.5)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################

'''
#had to use this for debugging
train_dataset = data_utils.Subset(train_dataset, range(500))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                batch_size=batch_size,
                shuffle=False)
'''

gen = MyGenerator()
dis = MyDiscriminator()

optimizergen = torch.optim.Adam(gen.parameters(), lr=.0002)
optimizerdis = torch.optim.Adam(dis.parameters(), lr=.0002)

#binary cross entropy for this problem
criteriondis = nn.BCELoss()
criteriongen = nn.BCELoss()


random_digits = torch.randn(5, 128)

train_loss_dis_epoch = []
train_loss_gen_epoch = []

epochs = 2
for epoch in range(epochs):

    train_loss_dis = 0
    train_loss_gen = 0
    for i, (images, _) in enumerate(train_loader):
        #flatten
        batch_size = images.size(0)
        images = images.view(-1, 28*28)

        #reset discriminator gradients
        optimizerdis.zero_grad()
        
        #generate fake images
        z = torch.randn(batch_size, 128)
        fake_images = gen.forward(z)

        #generate labels
        real_labels = torch.zeros(batch_size, 1)
        fake_labels = torch.ones(batch_size, 1)

        #discrim loss
        outputs = dis.forward(images)
        dis_loss_real = criteriondis(outputs,real_labels)

        outputs = dis.forward(fake_images)
        dis_loss_fake = criteriondis(outputs,fake_labels)

        dis_loss_tot = dis_loss_real + dis_loss_fake

        #update discriminator gradients
        dis_loss_tot.backward()
        optimizerdis.step()

        train_loss_dis += dis_loss_tot

        #since k=1 train generator
        optimizergen.zero_grad()
        z = torch.randn(batch_size, 128)
        fake_images = gen.forward(z)
        outputs = dis.forward(fake_images)

        #flip labels when evaluating loss
        gen_loss = criteriongen(outputs, real_labels)

        #optimize
        gen_loss.backward()
        optimizergen.step()

        train_loss_gen += gen_loss
    
    train_loss_dis_epoch.append(train_loss_dis.detach().numpy())
    train_loss_gen_epoch.append(train_loss_gen.detach().numpy())

    generated_digits = gen.forward(random_digits).detach().numpy()
    generated_digits = generated_digits.reshape(-1,28,28)

    # Plot the images
    fig, ax = plt.subplots(1, 5)
    for i in range(5):
        ax[i].imshow(generated_digits[i], cmap='gray')
        ax[i].axis('off')
    
    #save the 5 generatorated images to png
    plt.savefig(f'epoch_{epoch+1}.png')
    
    #print the loss
    print('Epoch [{}/{}], Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'
        .format(epoch+1, epochs, train_loss_dis, train_loss_gen))

#loss plot
x_values = range(1,len(train_loss_gen_epoch)+1)
plt.plot(x_values, train_loss_gen_epoch, label='Generator Loss')
plt.plot(x_values,train_loss_dis_epoch, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#general final images
generated_digits_final = gen.forward(random_digits).detach().numpy()
generated_digits_final = generated_digits.reshape(-1,28,28)

# Plot the final images
fig, ax = plt.subplots(1, 5)
for i in range(5):
    ax[i].imshow(generated_digits_final[i], cmap='gray')
    ax[i].axis('off')
plt.savefig(f'final_image.png')