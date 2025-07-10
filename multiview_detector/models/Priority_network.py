import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightPriorityNetwork(nn.Module):
    def __init__(self, num_cam, tau_1, channel):
        super(LightweightPriorityNetwork, self).__init__()
        self.num_cam = num_cam
        self.tau_1 = tau_1
        self.channel = channel

        # Global average pooling to reduce each feature map to a single value
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Output size [B, C, 1, 1]

        # Fully connected layer to map pooled features to a single priority score
        self.fc = nn.Linear(self.channel, 1)  # Map channels to priority

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Global pooling to reduce spatial dimensions to [B, num_cam * (tau_1 + 1), channel]
        pooled_features = self.global_pool(x).view(B, self.num_cam * (self.tau_1 + 1), self.channel)
        
        # Normalize features between 0 and 1 to account for object detection distribution
        normalized_features = F.normalize(pooled_features, p=2, dim=2)  # Normalize along the channel dimension
        
        # Compute priority by inverting the normalized feature values (low values mean higher priority)
        inverted_priority = 1 - normalized_features  # Invert, so lower values get higher priority
        
        # Pass through fully connected layer to get priority scores
        priority_vector = self.fc(inverted_priority).view(B, -1)  # Output shape [B, num_cam * (tau_1 + 1)]
        
        # Apply softmax to convert to probabilities (higher values mean higher priority)
        priority_vector = F.softmax(priority_vector, dim=1)  # Ensure probabilistic output
        
        return priority_vector
