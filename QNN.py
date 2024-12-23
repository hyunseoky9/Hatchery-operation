import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, StepLR

# Define model
class QNN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, hidden_num, learning_rate, state_min, state_max):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.hidden_num = hidden_num
        self.learning_rate = learning_rate

        # normalization parameters
        self.state_min = state_min
        self.state_max = state_max
        # Constructing the layers dynamically
        layers = [nn.Linear(state_size, hidden_size), nn.ReLU()]
        for _ in range(self.hidden_num - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, action_size))
        # Creating the Sequential module
        self.linear_relu_stack = nn.Sequential(*layers)

        # loss and optimizer
        self.loss_fn = nn.MSELoss()
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9997)  # Exponential decay
        #self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)  # Halve LR every 10 steps

        
    def forward(self, x):
        #x_norm = self.normalize(x)
        logits = self.linear_relu_stack(x)
        return logits

