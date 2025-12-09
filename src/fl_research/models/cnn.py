"""
Convolutional Neural Network Architectures
===========================================

CNN models for image classification in federated learning.
All models use GroupNorm for Opacus compatibility.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST/Fashion-MNIST (28x28 grayscale images).
    
    Architecture:
        Conv2d(1, 16, 5) -> MaxPool -> Conv2d(16, 32, 5) -> MaxPool -> FC(512, 128) -> FC(128, 10)
    
    Args:
        num_classes: Number of output classes (default: 10)
        in_channels: Number of input channels (default: 1 for grayscale)
    
    Example:
        >>> model = SimpleCNN(num_classes=10)
        >>> x = torch.randn(1, 1, 28, 28)
        >>> output = model(x)
        >>> output.shape
        torch.Size([1, 10])
    """
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CIFAR10CNN(nn.Module):
    """
    CNN for CIFAR-10 (32x32 RGB images).
    
    Architecture:
        Conv2d(3, 32, 3) -> MaxPool -> Conv2d(32, 64, 3) -> MaxPool -> 
        Conv2d(64, 64, 3) -> MaxPool -> FC(64*4*4, 256) -> FC(256, 10)
    
    Note: Uses GroupNorm for Opacus compatibility when dp_compatible=True.
    
    Args:
        num_classes: Number of output classes (default: 10)
        dp_compatible: If True, use GroupNorm instead of BatchNorm (default: True)
        dropout: Dropout probability (default: 0.25)
    
    Example:
        >>> model = CIFAR10CNN(num_classes=10)
        >>> x = torch.randn(1, 3, 32, 32)
        >>> output = model(x)
        >>> output.shape
        torch.Size([1, 10])
    """
    
    def __init__(
        self, 
        num_classes: int = 10, 
        dp_compatible: bool = True,
        dropout: float = 0.25
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dp_compatible = dp_compatible
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Use GroupNorm for DP compatibility (BatchNorm not supported by Opacus)
        if dp_compatible:
            self.norm1 = nn.GroupNorm(8, 32)
            self.norm2 = nn.GroupNorm(8, 64)
            self.norm3 = nn.GroupNorm(8, 64)
        else:
            self.norm1 = nn.BatchNorm2d(32)
            self.norm2 = nn.BatchNorm2d(64)
            self.norm3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CIFAR10CNNOpacus(nn.Module):
    """
    Opacus-compatible CNN for CIFAR-10 with GroupNorm.
    
    This is a simplified version specifically designed for DP training.
    Uses GroupNorm throughout and avoids any Opacus-incompatible layers.
    
    Args:
        num_classes: Number of output classes (default: 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ResNetBlock(nn.Module):
    """Basic ResNet block with optional GroupNorm for DP compatibility."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        dp_compatible: bool = True
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        
        if dp_compatible:
            self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
            self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        else:
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if dp_compatible:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(min(8, out_channels), out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetSmall(nn.Module):
    """
    Small ResNet variant for FL experiments.
    
    Designed for CIFAR-10 with reduced parameters for faster FL training.
    
    Args:
        num_classes: Number of output classes (default: 10)
        dp_compatible: If True, use GroupNorm (default: True)
    """
    
    def __init__(self, num_classes: int = 10, dp_compatible: bool = True):
        super().__init__()
        self.in_channels = 16
        self.dp_compatible = dp_compatible
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        if dp_compatible:
            self.norm1 = nn.GroupNorm(4, 16)
        else:
            self.norm1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, s, self.dp_compatible))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
