import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, StepLR

class A3CNN(nn.Module):
    def __init__(self, state_size, contaction, action_size, hidden_size, hidden_num, lstm, lstm_num, normalize, state_min, state_max):
        super().__init__()
        self.action_size = action_size # number of actions if discrete action space, number of parameters for a continuous distribution if continuous action space
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.hidden_num = hidden_num
        self.contaction = contaction # 1= continuous, 0= discrete
        self.normalization = normalize
        self.state_min = state_min
        self.state_max = state_max
        self.lstm = lstm # 1= use LSTM, 0= not use LSTM
        self.lstm_num = lstm_num # number of LSTM cells in a layer.
        self.type = 'A3C'
        layers = [nn.Linear(state_size, hidden_size), nn.ReLU()]
        for _ in range(self.hidden_num - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        if lstm:
            # Add LSTM layer
            self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=lstm_num, batch_first=True)

            # Separate policy and value heads after LSTM
            self.policy_head = nn.Linear(lstm_num, action_size)
            self.value_head = nn.Linear(lstm_num, 1)
        else:
            # No LSTM: Separate policy and value heads directly
            self.policy_head = nn.Linear(hidden_size, action_size)
            self.value_head = nn.Linear(hidden_size, 1)

        # Creating the Sequential module
        self.stack = nn.Sequential(*layers)

    def forward(self, x, hidden_state=None):
        if self.normalization:
            x = self.normalize(x)
        x = self.stack(x)
        if self.lstm:
            if len(x.shape) == 2:
                x = x.unsqueeze(0) # add batch dimension 
            
            if hidden_state is None:
                batch_size = x.size(0)
                hidden_state = (torch.zeros(1, batch_size, self.lstm_num),
                                     torch.zeros(1, batch_size, self.lstm_num))
            # pass through LSTM layer
            x, hidden_state = self.lstm_layer(x,hidden_state) # x shape: 
            x = x[:, -1, :]  # Extract last time step

        logits = self.policy_head(x)
        if self.contaction == 0:
            policy = nn.functional.softmax(logits, dim=-1)
        else:
            policy = logits

        value = self.value_head(x)
        return policy, value, hidden_state if self.lstm else None

    
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
