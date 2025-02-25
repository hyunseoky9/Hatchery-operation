import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from NoisyLinear import NoisyLinear

# Define model
class DuelQNN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size_shared, hidden_size_split, hidden_num_shared, hidden_num_split, learning_rate, state_min, state_max, lrdecayrate, noisy, distributional, atomn, Vmin, Vmax, normalize,fstack):
        super().__init__()
        # architecture parameters
        self.action_size = action_size
        self.state_size = state_size
        self.hidden_size_shared = hidden_size_shared
        self.hidden_size_split = hidden_size_split
        self.hidden_num_shared = hidden_num_shared
        self.hidden_num_split = hidden_num_split
        self.learning_rate = learning_rate
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
        ## shared layers
        if noisy:
            shared_layers = [NoisyLinear(state_size, hidden_size_shared), nn.ReLU()]
            for _ in range(self.hidden_num_shared - 1):
                shared_layers.append(NoisyLinear(hidden_size_shared, hidden_size_shared))
                shared_layers.append(nn.ReLU())

            ## value layers
            if hidden_num_split > 0:
                value_layers = [NoisyLinear(hidden_size_shared, hidden_size_split), nn.ReLU()]
                for _ in range(self.hidden_num_split - 1):
                    value_layers.append(NoisyLinear(hidden_size_split, hidden_size_split))
                    value_layers.append(nn.ReLU())
                value_layers.append(NoisyLinear(hidden_size_split, 1))
            else:
                value_layers = [NoisyLinear(hidden_size_shared, 1)]
            ## advantage layers
            if hidden_num_split > 0:
                advantage_layers = [NoisyLinear(hidden_size_shared, hidden_size_split), nn.ReLU()]
                for _ in range(self.hidden_num_split - 1):
                    advantage_layers.append(NoisyLinear(hidden_size_split, hidden_size_split))
                    advantage_layers.append(nn.ReLU())
                advantage_layers.append(NoisyLinear(hidden_size_split, action_size*self.atomn))
            else:
                advantage_layers = [NoisyLinear(hidden_size_shared, action_size*self.atomn)]
        else:
            shared_layers = [nn.Linear(state_size, hidden_size_shared), nn.ReLU()]
            for _ in range(self.hidden_num_shared - 1):
                shared_layers.append(nn.Linear(hidden_size_shared, hidden_size_shared))
                shared_layers.append(nn.ReLU())

            ## value layers
            if hidden_num_split > 0:
                value_layers = [nn.Linear(hidden_size_shared, hidden_size_split), nn.ReLU()]
                for _ in range(self.hidden_num_split - 1):
                    value_layers.append(nn.Linear(hidden_size_split, hidden_size_split))
                    value_layers.append(nn.ReLU())
                value_layers.append(nn.Linear(hidden_size_split, 1))
            else:
                value_layers = [nn.Linear(hidden_size_shared, 1)]
            ## advantage layers
            if hidden_num_split > 0:
                advantage_layers = [nn.Linear(hidden_size_shared, hidden_size_split), nn.ReLU()]
                for _ in range(self.hidden_num_split - 1):
                    advantage_layers.append(nn.Linear(hidden_size_split, hidden_size_split))
                    advantage_layers.append(nn.ReLU())
                advantage_layers.append(nn.Linear(hidden_size_split, action_size*self.atomn))
            else:
                advantage_layers = [nn.Linear(hidden_size_shared, action_size*self.atomn)]

        # Creating the Sequential module
        self.shared_linear_relu_stack = nn.Sequential(*shared_layers)
        self.value_linear_relu_stack = nn.Sequential(*value_layers)
        self.advantage_linear_relu_stack = nn.Sequential(*advantage_layers)

        # loss and optimizer
        if distributional: 
            self.loss_fn = nn.CrossEntropyLoss()
        else:
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
        shared_output = self.shared_linear_relu_stack(x)
        value = self.value_linear_relu_stack(shared_output)
        advantage = self.advantage_linear_relu_stack(shared_output)
        logits = value + (advantage - advantage.mean(dim=1, keepdim=True))
        #logits = value + (advantage - advantage.mean())       
        if self.distributional:
            logits = logits.view(-1, self.action_size, self.atomn)  # Reshape for actions and atoms
            probabilities = torch.softmax(logits, dim=-1)  # Apply softmax across the atoms
            return probabilities
        else:

            return logits


    def disable_noise(self):
        # Disable noise for all NoisyLinear layers (for validation)
        for layer in self.shared_linear_relu_stack:
            if isinstance(layer, NoisyLinear):
                layer.use_noise = False
        for layer in self.value_linear_relu_stack:
            if isinstance(layer, NoisyLinear):
                layer.use_noise = False
        for layer in self.advantage_linear_relu_stack:
            if isinstance(layer, NoisyLinear):
                layer.use_noise = False

    def enable_noise(self):
        # Enable noise for all NoisyLinear layers
        for layer in self.shared_linear_relu_stack:
            if isinstance(layer, NoisyLinear):
                layer.use_noise = True
        for layer in self.value_linear_relu_stack:
            if isinstance(layer, NoisyLinear):
                layer.use_noise = True
        for layer in self.advantage_linear_relu_stack:
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
