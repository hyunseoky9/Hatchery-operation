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

    def train_model(self, data, device):
        self.train()
        for batch, (states, actions, targets) in enumerate(data):
            states, actions, targets = states.to(device), actions.to(device), targets.to(device)

            # Compute prediction error
            loss = self.compute_loss(states, actions, targets)
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def compute_loss(self, states, actions, targetQs): 
        """
        Compute the loss and perform a backward pass.
        
        Parameters:
            states (torch.Tensor): Input states.
            actions (torch.Tensor): Actions taken (as indices).
            targetQs (torch.Tensor): Target Q values.
        """
        q_values = self(states) # Forward pass
        selected_q_values = q_values.gather(1, actions).squeeze(1) # Get Q-values for the selected actions
        loss = self.loss_fn(selected_q_values, targetQs) # Compute the loss
        return loss
    
    def test_model(self, reachable_states, reachable_actions, Qopt, device):
        """
        If there is a optimal Q calculate (perhaps from value iteration) MSE loss compared to the optimal Q.
        """
        self.eval()
        with torch.no_grad():
            testloss = self.compute_loss(reachable_states, reachable_actions, Qopt).item()
        return testloss
        #print(f"Test Error Avg loss: {test_loss:>8f}\n")

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
        