import numpy as np

import torch
import torch.nn as nn

#basic MLP discriminator with dropout to reduce fit
class MyDiscriminator(nn.Module):
    def __init__(self):
        super(MyDiscriminator, self).__init__()
        self.discmodel = nn.Sequential(
            nn.Linear(784, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discmodel(x)
        return x
