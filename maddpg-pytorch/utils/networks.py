import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

        if norm_in:  # normalize inputs
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
        # h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class CNNNetwork(nn.Module):
    """
    CNN network for image-based environments (like Atari)
    Processes image observations through convolutional layers then MLP
    """
    def __init__(self, input_shape, out_dim, hidden_dim=512, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_shape (tuple): Shape of input images (height, width, channels) e.g., (210, 160, 3)
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions in MLP layers
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
            constrain_out (bool): Whether to constrain output with tanh
            norm_in (bool): Whether to apply batch normalization to MLP inputs
            discrete_action (bool): Whether actions are discrete
        """
        super(CNNNetwork, self).__init__()
        
        # Store input shape for preprocessing
        self.input_shape = input_shape
        h, w, c = input_shape
        
        # Convolutional layers - designed for Atari-style inputs
        # Input normalization: scale from [0, 255] to [0, 1]
        self.conv_layers = nn.Sequential(
            # First conv layer: reduce spatial dimensions significantly
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=2),  # Output: (32, ~52, ~40)
            nn.ReLU(),
            # Second conv layer: further feature extraction
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, ~26, ~20)
            nn.ReLU(),
            # Third conv layer: final feature extraction
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, ~26, ~20)
            nn.ReLU()
        )
        
        # Calculate the flattened size after conv layers
        # We need to do a forward pass to determine this
        with torch.no_grad():
            # Create dummy input with correct shape (batch_size=1, channels, height, width)
            dummy_input = torch.zeros(1, c, h, w)
            conv_output = self.conv_layers(dummy_input)
            self.conv_output_size = conv_output.numel()
        
        # Batch normalization for MLP input (similar to MLPNetwork)
        if norm_in:
            self.in_fn = nn.BatchNorm1d(self.conv_output_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        
        # MLP layers after flattening
        self.fc1 = nn.Linear(self.conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        
        # Output function setup
        if constrain_out and not discrete_action:
            # Initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x
    
    def preprocess_image(self, x):
        """
        Preprocess image observations:
        - Convert from (batch, height, width, channels) to (batch, channels, height, width)
        - Normalize from [0, 255] to [0, 1]
        """
        if x.dim() == 3:  # Single image (height, width, channels)
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Convert from (batch, height, width, channels) to (batch, channels, height, width)
        if x.shape[-1] in [1, 3, 12]:  # Last dimension is channels. 12 occurs when 4 rgb frames are stacked
            x = x.permute(0, 3, 1, 2)
        
        # Normalize from [0, 255] to [0, 1]
        x = x.float() / 255.0
        
        return x
    
    def forward(self, X):
        """
        Inputs:
            X (PyTorch Tensor): Batch of image observations
                               Shape: (batch_size, height, width, channels) or 
                                      (batch_size, channels, height, width)
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # Preprocess images
        X = self.preprocess_image(X)
        
        # Pass through convolutional layers
        conv_out = self.conv_layers(X)
        
        # Flatten for MLP
        flattened = conv_out.reshape(conv_out.size(0), -1)
        
        # Pass through MLP layers with batch normalization
        h1 = self.nonlin(self.fc1(self.in_fn(flattened)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        
        return out


class MultiAgentCriticNetwork(nn.Module):
    """
    Multi-agent critic network for MADDPG with image observations.
    Processes each agent's image observations through separate CNN branches,
    then combines all features with actions for centralized value estimation.
    """
    def __init__(self, num_agents, obs_shapes, total_action_dim, hidden_dim=512, 
                 nonlin=F.relu, norm_in=True, obs_types=None):
        """
        Inputs:
            num_agents (int): Number of agents in the environment
            obs_shapes (list): List of observation shapes for each agent
                              For images: (height, width, channels)
                              For vectors: (dim,)
            total_action_dim (int): Total dimension of all agents' actions combined
            hidden_dim (int): Number of hidden dimensions in MLP layers
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
            norm_in (bool): Whether to apply batch normalization
            obs_types (list): List of observation types for each agent ('image' or 'vector')
                             If None, infers from obs_shapes
        """
        super(MultiAgentCriticNetwork, self).__init__()
        
        self.num_agents = num_agents
        self.obs_shapes = obs_shapes
        self.total_action_dim = total_action_dim
        self.obs_types = obs_types or self._infer_obs_types(obs_shapes)
        
        # Create feature extractors for each agent
        self.feature_extractors = nn.ModuleList()
        feature_dims = []
        
        for i in range(num_agents):
            if self.obs_types[i] == 'image':
                # CNN feature extractor for image observations
                h, w, c = obs_shapes[i]
                
                # Convolutional layers (same as CNNNetwork)
                conv_layers = nn.Sequential(
                    nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
                )
                
                # Calculate conv output size
                with torch.no_grad():
                    dummy_input = torch.zeros(1, c, h, w)
                    conv_output = conv_layers(dummy_input)
                    conv_output_size = conv_output.numel()
                
                # Feature extractor with conv + linear layers
                feature_extractor = nn.Sequential(
                    conv_layers,
                    nn.Flatten(),
                    nn.Linear(conv_output_size, hidden_dim // 2),
                    nn.ReLU()
                )
                feature_dims.append(hidden_dim // 2)
                
            else:  # vector observation
                # MLP feature extractor for vector observations
                obs_dim = obs_shapes[i][0] if isinstance(obs_shapes[i], tuple) else obs_shapes[i]
                feature_extractor = nn.Sequential(
                    nn.Linear(obs_dim, hidden_dim // 2),
                    nonlin
                )
                feature_dims.append(hidden_dim // 2)
            
            self.feature_extractors.append(feature_extractor)
        
        # Calculate total input dimension for the final MLP
        total_obs_features = sum(feature_dims)
        total_input_dim = total_obs_features + self.total_action_dim
        
        # Batch normalization for combined features
        if norm_in:
            self.in_fn = nn.BatchNorm1d(total_input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        
        # Final MLP layers for value estimation
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output single Q-value
        self.nonlin = nonlin
        
        # Initialize output layer with small weights
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def _infer_obs_types(self, obs_shapes):
        """Infer observation types from shapes"""
        obs_types = []
        for shape in obs_shapes:
            if isinstance(shape, tuple) and len(shape) == 3:
                obs_types.append('image')  # (height, width, channels)
            else:
                obs_types.append('vector')  # 1D vector
        return obs_types
    
    def preprocess_image(self, x):
        """Same preprocessing as CNNNetwork"""
        if x.dim() == 3:  # Single image
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Convert from (batch, height, width, channels) to (batch, channels, height, width)
        if x.shape[-1] in [1, 3, 12]:  # Last dimension is channels. 12 occurs when 4 rgb frames are stacked
            x = x.permute(0, 3, 1, 2)
        
        # Normalize from [0, 255] to [0, 1]
        x = x.float() / 255.0
        
        return x
    
    def forward(self, observations, actions):
        """
        Inputs:
            observations (list): List of observation tensors for each agent
            actions (list): List of action tensors for each agent
        Outputs:
            Q-value (PyTorch Tensor): Centralized Q-value estimation
        """
        batch_size = observations[0].size(0)
        
        # Extract features from each agent's observations
        agent_features = []
        for i in range(self.num_agents):
            obs = observations[i]
            
            if self.obs_types[i] == 'image':
                # Preprocess images
                obs = self.preprocess_image(obs)
            
            # Extract features
            features = self.feature_extractors[i](obs)
            agent_features.append(features)
        # Concatenate all observation features and actions
        obs_features = torch.cat(agent_features, dim=1)
        action_tensors = torch.cat(actions, dim=1)
        combined_input = torch.cat([obs_features, action_tensors], dim=1)
        
        # Pass through final MLP with batch normalization
        h1 = self.nonlin(self.fc1(self.in_fn(combined_input)))
        h2 = self.nonlin(self.fc2(h1))
        q_value = self.fc3(h2)
        return q_value