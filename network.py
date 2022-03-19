import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Network(nn.Module):
    def __init__(self, input_shape,number_of_actions, n_hidden_layers=3, n_hidden_nodes=64,n_last_nodes=32,learning_rate=0.001, bias=False, device='cuda'):
        super(Network, self).__init__()
        # get the inputs
        self.device = device
        self.n_inputs = input_shape
        self.n_outputs = number_of_actions
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.bias = bias
        self.action_space = np.arange(self.n_outputs)
        # define body of the network
        self.layers = OrderedDict()
        self.n_layers = 2 * self.n_hidden_layers
        for i in range(self.n_layers):
            if i == 0 and self.n_hidden_layers != 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_inputs,
                    self.n_hidden_nodes,
                    bias=self.bias)
            elif i % 2 == 0 and i != 0:
                self.layers[str(i)] = nn.Linear(
                    self.n_hidden_nodes,
                    self.n_hidden_nodes,
                    bias=self.bias)
            else:
                self.layers[str(i)] = nn.ReLU()
        self.body = nn.Sequential(self.layers)
        # define policy head
        self.policy = nn.Sequential(
            nn.Linear(self.n_hidden_nodes,
                      n_last_nodes,
                      bias=self.bias),
            nn.ReLU(),
            nn.Linear(n_last_nodes,
                      self.n_outputs,
                      bias=self.bias))
        # define value head
        self.value = nn.Sequential(
            nn.Linear(self.n_hidden_nodes,
                      n_last_nodes,
                      bias=self.bias),
            nn.ReLU(),
            nn.Linear(n_last_nodes,
                      1,
                      bias=self.bias))
        # other settings
        if self.device == 'cuda':
            self.body.cuda()
            self.policy.cuda()
            self.value.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.learning_rate,weight_decay=1e-5)

    # define functions to get outputs from network
    def predict(self, state):
        body_output = self.get_body_output(state)
        probs = F.softmax(self.policy(body_output), dim=-1)
        return probs, self.value(body_output)

    def get_action(self, state):
        probs = self.predict(state)[0].detach().numpy()
        action = np.random.choice(self.action_space, p=probs)
        return action

    def get_log_probs(self, state):
        body_output = self.get_body_output(state)
        logprobs = F.log_softmax(self.policy(body_output), dim=-1)
        return logprobs

    def get_body_output(self, state):
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.body(torch.flatten(state_t))
    
