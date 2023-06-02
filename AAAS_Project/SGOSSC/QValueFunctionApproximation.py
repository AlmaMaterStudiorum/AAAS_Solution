import torch

import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# https://www.geeksforgeeks.org/python-oops-concepts/
# # parent class :  nn.Module https://pytorch.org/docs/stable/generated/torch.nn.Module.html
#                   Base class for all neural network modules.
#                   Your models should also subclass this class.



class QValueFunctionApproximation(nn.Module):

    # default constructor
    # Instance attribute
    # observation_space_size : definizione di stato (ingresso)
    # action_space_size      : definizione di azioni (ingresso)
    # uscita                 : Q VALUE sempre 1 (appartiene ad R)
    # hidden_size : dimensione del layer hidden
    def __init__(self, observation_space_size , action_space_size, hidden_layers, learning_rate=3e-4):
        super(QValueFunctionApproximation, self).__init__()
        # super().__init__()
        if False :

            # activationfunction = nn.Sigmoid()
            activationfunction = nn.ReLU()

            self.hidden_layers = hidden_layers

            layers = []

            num_inputs = observation_space_size 
            num_output = action_space_size

            layers.append( nn.Linear(num_inputs, hidden_layers[0]))
            layers.append( activationfunction )

            for i in range(0,len(hidden_layers) -1):
                layers.append( nn.Linear(hidden_layers[i], hidden_layers[i+1]) )
                layers.append( activationfunction )

            layers.append( nn.Linear(hidden_layers[-1], num_output) )


            # If you have a model with lots of layers, you can create a list first and then use the * operator to expand the list into positional arguments
            # https://stackoverflow.com/questions/46141690/how-do-i-write-a-pytorch-sequential-model
            self.model = nn.Sequential(*layers)
            for param in self.model.parameters():
                param.requires_grad = True

        #self.model =   nn.Sequential(
        #                  nn.Linear(observation_space_size, action_space_size),
        #                  nn.ReLU(),
        #                  nn.Linear(action_space_size, 1),
        #                  nn.ReLU()
        #                  )

        # It appears that grad doen's work with Relu
        self.model =   nn.Sequential(
                    nn.Linear(observation_space_size, action_space_size*(2**2)),
                    nn.Linear(action_space_size*(2**2), action_space_size*(2**1)),
                    nn.Linear(action_space_size*(2**1), action_space_size),
                    nn.Linear(action_space_size, 1)
                    )
        #for param in self.model.parameters():
        #    param.requires_grad = True

    
    def forward(self, state):
        # https://discuss.pytorch.org/t/why-do-i-get-typeerror-expected-np-ndarray-got-numpy-ndarray-when-i-use-torch-from-numpy-function/37525/12

        # type of state is array
        # tensor = torch.tensor(state, dtype=torch.float)
        # tensor = torch.from_numpy(state)
        # type of state is tensor
        # tensor = tensor.float()
        tensor = torch.Tensor(state)
        tensor = tensor.float()
        # https://sebarnold.net/tutorials/beginner/pytorch_with_examples.html#pytorch-variables-and-autograd
        # https://www.geeksforgeeks.org/variables-and-autograd-in-pytorch/

        # tensor.requires_grad = True
        # print(tensor.is_leaf)


        output =  self.model(tensor)
        # output = output.clone().detach().requires_grad_(True)

        # print(output.is_leaf)

        # qvfa_output dim 1
        return output


