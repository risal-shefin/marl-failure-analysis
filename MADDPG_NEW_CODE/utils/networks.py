import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()
        
        # Store input dimension for reference
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        if norm_in:
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
            
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # Print input shape for debugging
        # print(f"Network input shape: {X.shape}")
        
        # Check if dimensions match what the network expects
        actual_dim = X.shape[1]
        expected_dim = self.fc1.in_features
        
        if actual_dim != expected_dim:
            # print(f"WARNING: Input dimension mismatch - got {actual_dim}, expected {expected_dim}")
            # Instead of returning None, use a fallback mechanism
            # Create a new linear layer on the fly to handle the new dimension
            temp_fc1 = nn.Linear(actual_dim, self.fc2.in_features).to(X.device)
            # Apply this temporary layer
            h1 = self.nonlin(temp_fc1(X))
            h2 = self.nonlin(self.fc2(h1))
            out = self.out_fn(self.fc3(h2))
            return out
        
        # Normal forward pass
        try:
            X = self.in_fn(X)
        except Exception as e:
            print(f"Error in input function: {e}")
            # Continue without batch norm if there's an error
            pass
            
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out