import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# Define model
class RQNN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, hidden_num, lstm_num, lstm_layers, batch_size, seql, learning_rate, state_min, state_max, lrdecayrate, distributional, atomn, Vmin, Vmax, normalize):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.hidden_num = hidden_num
        self.lstm_num = lstm_num # number of LSTM cells in a layer.
        self.lstm_layers = lstm_layers # number of LSTM layers
        self.batch_size = batch_size # mini-batch size for the LSTM
        self.seql = seql # 
        self.learning_rate = learning_rate
        self.distributional = distributional
        self.normalization = normalize
        if distributional:
            self.atomn = atomn
            self.z = torch.linspace(Vmin, Vmax, atomn)
        else:
            self.atomn = 1

        # normalization parameters
        self.state_min = state_min
        self.state_max = state_max

        # Constructing the layers dynamically
        layers = [nn.Linear(state_size, hidden_size), nn.ReLU()]
        for _ in range(self.hidden_num - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        # Creating the Sequential module
        self.stack = nn.Sequential(*layers)

        # Add LSTM layer
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=lstm_num, num_layers=lstm_layers, batch_first=True)
        # Output layer
        self.head = nn.Linear(lstm_num, action_size*self.atomn)

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
    def forward(self, x, training=True ,hidden=None, lengths=None):
        if self.normalization:
            x = self.normalize(x)


        x = self.stack(x) # pass through the FF stack
        if training: # if training data coming in as a batch.
            if hidden is None: # initialize hidden state
                hidden = (torch.zeros(self.lstm_layers, self.batch_size, self.lstm_num, device=x.device, dtype=x.dtype),
                                        torch.zeros(self.lstm_layers, self.batch_size, self.lstm_num, device=x.device, dtype=x.dtype))
            # Pack the feedforward output
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            x, hidden = self.lstm_layer(x,hidden) # pass through LSTM layer
            x, unpacked_lengths = pad_packed_sequence(x, batch_first=True)
            logits = self.head(x)

        else: # if not training (simulation or evaluation), then x is a single state
            if hidden is None: # initialize hidden state
                hidden = (torch.zeros(self.lstm_layers, 1, self.lstm_num, device=x.device, dtype=x.dtype),
                                        torch.zeros(self.lstm_layers, 1, self.lstm_num, device=x.device, dtype=x.dtype))

            if len(x.shape) == 2: # if input is a single state (most likely when choosing an action during evaluation or online simulation)
                x = x.unsqueeze(0) # add batch dimension 
            x, hidden = self.lstm_layer(x,hidden) # pass through LSTM layer
            logits = self.head(x)

        if self.distributional:
            logits = logits.view(-1, self.action_size, self.atomn)  # Reshape for actions and atoms
            probabilities = torch.softmax(logits, dim=-1)  # Apply softmax across the atoms
            if training:
                return probabilities, hidden
        else:
            return logits, hidden

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
    

