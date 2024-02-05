import numpy as np

import torch
import torch.nn as nn

class MyAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate, max_epochs):
        '''
        input_size: [int], feature dimension 
        hidden_size: number of hidden nodes in the hidden layer
        output_size: number of classes in the dataset, 
        learning_rate: learning rate for gradient descent,
        max_epochs: maximum number of epochs to run gradient descent
        '''
        #initialize autoencoder
        super(MyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size, bias=True),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size, bias=True),
            nn.Sigmoid(),
        )

        self.output_size = output_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def forward(self, x):
        ''' Function to do the forward pass with images x '''
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def fit(self, train_loader):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.critereon = nn.MSELoss()
        plot_loss = []

        for i in range(self.max_epochs):
            train_loss = 0

            for j,(images, _) in enumerate(train_loader):
                images = images.view(-1, self.input_size)
                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = self.critereon(outputs, images)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            
            print('Epoch [{}/{}], Train Loss: {:.4f}'.format(i+1, self.max_epochs, train_loss))
            plot_loss.append(train_loss)

        return plot_loss
    

    def encode(self, test_loader):

        encoded_samples = []
        train_labels = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.view(-1, self.input_size)
                encoded = self.encoder(images)
                encoded_samples.append(encoded.tolist())

                train_labels.append(labels)
        
        return encoded_samples, train_labels
