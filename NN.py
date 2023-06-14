# We use torch to store the numerical datas as tensors
import torch
# We import torch.nn which will use to make the weight and bias tensors
import torch.nn as nn
# We import the torch.nn.function to import the activation function
import torch.nn.functional as F
# We import this library to get stochastic gradient descent
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns



# Creating the neural network
class BasicNN(nn.Module):

    def __init__(self):
        # Initialization method for the parent class
        super().__init__()
        # now we will initialize the weight and bias vriable
        self.w00=nn.Parameter(torch.tensor(1.7),requires_grad=False)
        self.b00=nn.Parameter(torch.tensor(-0.85),requires_grad=False)
        self.w01=nn.Parameter(torch.tensor(-40.8),requires_grad=False)

        self.w10=nn.Parameter(torch.tensor(12.6),requires_grad=False)
        self.b10=nn.Parameter(torch.tensor(0.0),requires_grad=False)
        self.w11=nn.Parameter(torch.tensor(2.7),requires_grad=False)

        self.finalbias=nn.Parameter(torch.tensor(-16.),requires_grad=False)


# Training the neural network using backpropagation
class BasicNN_train(nn.Module):

    def __init__(self):
        # Initialization method for the parent class
        super().__init__()
        # now we will initialize the weight and bias vriable
        self.w00=nn.Parameter(torch.tensor(1.7),requires_grad=False)
        self.b00=nn.Parameter(torch.tensor(-0.85),requires_grad=False)
        self.w01=nn.Parameter(torch.tensor(-40.8),requires_grad=False)

        self.w10=nn.Parameter(torch.tensor(12.6),requires_grad=False)
        self.b10=nn.Parameter(torch.tensor(0.0),requires_grad=False)
        self.w11=nn.Parameter(torch.tensor(2.7),requires_grad=False)

        self.finalbias=nn.Parameter(torch.tensor(0.),requires_grad=True)
    

    def forward(self,input):
        input_to_top_relu=input*self.w00+self.b00
        top_relu=F.relu(input_to_top_relu)
        scaled_top_relu_output=top_relu*self.w01

        input_to_bottom_relu=input*self.w10+self.b10
        bottom_relu=F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output=bottom_relu*self.w11

        input_to_final_relu=scaled_top_relu_output+scaled_bottom_relu_output+self.finalbias
        output=F.relu(input_to_final_relu)

        return output
    

model = BasicNN_train()

input_doses=torch.linspace(start=0,end=1,steps=11)

outputvalues=model(input_doses)

print(outputvalues)

plt.plot(input_doses,outputvalues.detach(),color='green')
plt.ylabel('Effectiveness')
plt.xlabel('Dose')
        
plt.show()