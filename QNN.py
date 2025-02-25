import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from NoisyLinear import NoisyLinear

# Define model
class QNN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, hidden_num, learning_rate, state_min, state_max,lrdecayrate, noisy, distributional, atomn, Vmin, Vmax, normalize, fstack):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.hidden_num = hidden_num
        self.learning_rate = learning_rate
        self.noisy = noisy
        self.distributional = distributional
        self.normalization = normalize
        self.fstack = fstack
        if distributional:
            self.atomn = atomn
            self.z = torch.linspace(Vmin, Vmax, atomn)
        else:
            self.atomn = 1

        # normalization parameters
        self.state_min = state_min
        self.state_max = state_max

        # Constructing the layers dynamically
        if noisy:
            layers = [NoisyLinear(state_size, hidden_size), nn.ReLU()]
            for _ in range(self.hidden_num - 1):
                layers.append(NoisyLinear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            layers.append(NoisyLinear(hidden_size, action_size * self.atomn))
        else:
            layers = [nn.Linear(state_size, hidden_size), nn.ReLU()]
            for _ in range(self.hidden_num - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, action_size * self.atomn))

        # Creating the Sequential module
        self.linear_relu_stack = nn.Sequential(*layers)

        # loss and optimizer
        if self.distributional: # distributional
            self.loss_fn = nn.CrossEntropyLoss()
        else: # normal Q-learning
            self.loss_fn = nn.MSELoss()
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lrdecayrate)  # Exponential decay
        #self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)  # Halve LR every 10 steps

        
    def forward(self, x):
        if self.normalization:
            x = self.normalize(x)
        logits = self.linear_relu_stack(x)
        if self.distributional:
            logits = logits.view(-1, self.action_size, self.atomn)  # Reshape for actions and atoms
            probabilities = torch.softmax(logits, dim=-1)  # Apply softmax across the atoms
            return probabilities
        else:
            return logits

    def disable_noise(self):
        # Disable noise for all NoisyLinear layers (for validation)
        for layer in self.linear_relu_stack:
            if isinstance(layer, NoisyLinear):
                layer.use_noise = False

    def enable_noise(self):
        # Enable noise for all NoisyLinear layers
        for layer in self.linear_relu_stack:
            if isinstance(layer, NoisyLinear):
                layer.use_noise = True

    def normalize(self, state):
        """
        min-max normalization for discrete states.
        parmaeters: 
            states (torch.Tensor): Input states
            env (object): Environment object
        """
        # Normalize using broadcasting
        state_norm = (state - self.state_min) / (self.state_max - self.state_min)
        return state_norm
    

