import torch
from torch import nn
from torchvision.transforms import ToTensor

# Define model
class fnapproxto(nn.Module):
    def __init__(self, state_size=5, action_size=1, hidden_size=10, learning_rate=0.01):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        dropoutp = 0.1
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.loss_fn = nn.MSELoss()
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Add a learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def train_model(self, data, device):
        size = len(data)
        self.train()
        for batch, (X, y) in enumerate(data):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = self(X)
            loss = self.loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            #loss, current = loss.item(), (batch + 1) * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_model(self, data, device):
        num_batches = len(data)
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in data:
                X, y = X.to(device), y.to(device)
                pred = self(X)
                test_loss += self.loss_fn(pred, y).item()
        test_loss /= num_batches
        # Update the scheduler based on validation loss
        self.scheduler.step(test_loss)
        return test_loss
        #print(f"Test Error Avg loss: {test_loss:>8f}\n")

                    
            