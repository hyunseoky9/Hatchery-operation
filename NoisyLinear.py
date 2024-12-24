import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ExponentialLR, StepLR

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NoisyLinear, self).__init__()
        # Trainable parameters
        self.mu = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        # Noise buffers (not trainable)
        self.register_buffer('input_noise', torch.empty(in_features))
        self.register_buffer('output_noise', torch.empty(out_features))

        self.use_noise = True
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize mu and sigma following the NoisyNet paper
        nn.init.uniform_(self.mu, -1 / self.mu.size(1) ** 0.5, 1 / self.mu.size(1) ** 0.5)
        nn.init.constant_(self.sigma, 0.5 / self.sigma.size(1) ** 0.5)
        nn.init.uniform_(self.bias_mu, -1 / self.bias_mu.size(0) ** 0.5, 1 / self.bias_mu.size(0) ** 0.5)
        nn.init.constant_(self.bias_sigma, 0.5 / self.bias_sigma.size(0) ** 0.5)

    def forward(self, input):
        # Generate factorised noise
        if self.use_noise:
            self.input_noise.normal_()
            self.output_noise.normal_()

            # Apply f(x) = sgn(x) * sqrt(|x|)
            input_noise = self.input_noise.sign() * self.input_noise.abs().sqrt()
            output_noise = self.output_noise.sign() * self.output_noise.abs().sqrt()
            
            # Compute noise for weights and biases
            weight_noise = torch.ger(output_noise, input_noise)  # Outer product
            bias_noise = output_noise

            # Add noise to weights and biases
            noisy_weights = self.mu + self.sigma * weight_noise
            noisy_bias = self.bias_mu + self.bias_sigma * bias_noise
        else:
            noisy_weights = self.mu
            noisy_bias = self.bias_mu

        # Apply the noisy weights and biases
        return torch.addmm(noisy_bias, input, noisy_weights.t())        
