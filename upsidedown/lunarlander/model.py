from ray.tune import Trainable
from collections import namedtuple
import torch.nn.functional as F
import time 
from itertools import cycle
from gym.wrappers.frame_stack import FrameStack
import numpy as np
import random
import datetime
import torch
from torch import nn
from itertools import count


def swish(x, beta=1):                                                                                                                                                                                      
    return x * torch.sigmoid(beta * x)

class Behavior(nn.Module):
    def __init__(self, hidden_size, state_shape, num_actions, return_scale, horizon_scale):
        super(Behavior, self).__init__()
        self.return_scale = return_scale
        self.horizon_scale = horizon_scale
        # Extra action representation for "start" of episode action.
        self.emb_prev_action = torch.nn.Embedding(num_embeddings=num_actions+1, embedding_dim=hidden_size)
        self.fc_state = nn.Linear(state_shape[0] * state_shape[1], hidden_size)
        self.fc_dr = nn.Linear(1, hidden_size)
        self.fc_dh = nn.Linear(1, hidden_size)
        
        self.fc_one = nn.Linear(hidden_size, hidden_size)
        self.fc_two = nn.Linear(hidden_size, hidden_size)
        self.fc_three = nn.Linear(hidden_size, hidden_size)
        self.fc_four = nn.Linear(hidden_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)

    def forward(self, prev_action, state, dr, dh):
        state = state.view(state.shape[0], -1)

        output_prev_action = self.emb_prev_action(prev_action)
        output_state = self.fc_state(state)

        output_dr = self.fc_dr(dr * self.return_scale)
        output_dh = self.fc_dh(dh * self.horizon_scale)

        # output = torch.cat([output_dr, output_dh], 1)
        one = swish(self.fc_one(output_prev_action))
        two = swish(self.fc_two(output_state))
        three = swish(self.fc_three(output_dr))
        four = swish(self.fc_four(output_dh))

        output = torch.cat([one, two, three, four], 1)
        
        # sum1 = (output_prev_action + output_state)
        # sum2 = (output_dr + output_dh)
        # output = sum1 * sum2 # TODO: Is this a good way to combine these?
        
        output = swish(self.fc1(output))
        output = swish(self.fc2(output))
        output = self.fc3(output)
        return output


