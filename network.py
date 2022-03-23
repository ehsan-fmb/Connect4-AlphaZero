import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU,Conv2d,BatchNorm2d


class Network(nn.Module):
    def __init__(self,number_of_actions,learning_rate=0.001, bias=False, device='cuda'):
        super(Network, self).__init__()
        # get the inputs
        self.device = device
        self.n_outputs = number_of_actions
        self.learning_rate = learning_rate
        self.bias = bias
        self.action_space = np.arange(self.n_outputs)
        # define body of the network
        self.body = nn.Sequential(
            #first conv layer
            Conv2d(1, 32, kernel_size=4),
            BatchNorm2d(32),
            ReLU(inplace=True),
            # Second conv layer
            Conv2d(32, 64, kernel_size=2),
            BatchNorm2d(64),
            ReLU(inplace=True),
        )
        # define policy head
        self.policy = nn.Sequential(
            nn.Linear(384,
                      32,
                      bias=self.bias),
            nn.ReLU(),
            nn.Linear(32,
                      self.n_outputs,
                      bias=self.bias))
        # define value head
        self.value = nn.Sequential(
            nn.Linear(384,
                      32,
                      bias=self.bias),
            nn.ReLU(),
            nn.Linear(32,
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
        body_output = torch.flatten(self.get_body_output(state))
        print(body_output.shape)
        probs = F.softmax(self.policy(body_output), dim=-1)
        return probs, self.value(body_output)

    def get_action(self, state):
        probs = self.predict(state)[0].detach().numpy()
        action = np.random.choice(self.action_space, p=probs)
        return action

    def get_log_probs(self, state):
        body_output = torch.flatten(self.get_body_output(state))
        logprobs = F.log_softmax(self.policy(body_output), dim=-1)
        return logprobs

    def get_body_output(self, state):
        state=np.reshape(state,(1,1,state.shape[0],state.shape[1]))
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.body(state_t)
    
