import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, obs_dim, action_dim) -> None:
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(obs_dim[-1], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._conv2d_size_out(obs_dim)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def _conv2d_size_out(self, shape) -> int:
        """
        Computes the output size of a convolutional layer given an input size
        """
        state = torch.zeros(1, shape[2], shape[0], shape[1])
        o = self.conv1(state)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x) -> torch.Tensor:
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # BS x n_filters x H x W
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)