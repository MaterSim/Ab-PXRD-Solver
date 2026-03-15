"""
XRD Model Definitions for Prediction
Model definitions for XRD-based prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .formula_enhanced_features_v2 import batch_preprocess_formulas_enhanced_v2

# model_network

class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention mechanism to recalibrate channel-wise features."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),  # At least 4 neurons
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y

class EnhancedXRDResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()

        # Multi-scale convolutions to capture XRD peaks of different widths
        self.conv1_narrow = nn.Conv1d(in_channels, out_channels//2, kernel_size, stride,
                                     padding=kernel_size//2, bias=False)
        self.conv1_wide = nn.Conv1d(in_channels, out_channels//2, kernel_size+4, stride,
                                   padding=(kernel_size+4)//2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1,
                              padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual connection for better gradient flow
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.dropout = nn.Dropout(dropout)

        # Squeeze-and-Excitation attention for channel recalibration
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)

        # Fuse features from different convolutional scales
        narrow_features = self.conv1_narrow(x)
        wide_features = self.conv1_wide(x)
        out = torch.cat([narrow_features, wide_features], dim=1)

        out = F.relu(self.bn1(out))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # Apply channel attention
        out = self.se(out)

        out += residual
        out = F.relu(out)

        return out

class MultiKernelConv(nn.Module):
    """Combines multiple convolution kernels to capture XRD peaks of various widths."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        return x3 + x5 + x7

class ChannelAttention(nn.Module):
    """Channel attention mechanism to focus on the most informative feature channels."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class GlobalStdPool1d(nn.Module):
    """Global standard deviation pooling to summarize the distribution of XRD peaks."""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.std(x, dim=2)



class ImprovedXRDNetWithFormula(nn.Module):
    """
    Improved XRD network with optional formula (Bag-of-Elements) integration.
    """
    def __init__(self, input_dim=3500, num_classes=230, dropout_rate=0.2,
                 formula_dim=100, use_formula=True):
        super().__init__()

        self.use_formula = use_formula

        # XRD encoder (main convolutional stem)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )

        self.layer1 = self._make_layer(32, 64, 3, stride=2, dropout=dropout_rate * 0.3)
        self.layer2 = self._make_layer(64, 128, 3, stride=2, dropout=dropout_rate * 0.4)
        self.layer3 = self._make_layer(128, 256, 4, stride=2, dropout=dropout_rate * 0.5)
        self.layer4 = self._make_layer(256, 512, 3, stride=2, dropout=dropout_rate * 0.6)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Formula encoder (simple MLP)
        if use_formula:
            self.formula_encoder = nn.Sequential(
                nn.Linear(formula_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5)
            )
            # Feature dimension after fusion
            fused_dim = 512 * 2 + 128  # XRD(1024) + Formula(128)
        else:
            fused_dim = 512 * 2

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fused_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout):
        layers = []
        layers.append(EnhancedXRDResBlock(in_channels, out_channels, stride=stride, dropout=dropout))
        for _ in range(1, num_blocks):
            layers.append(EnhancedXRDResBlock(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, xrd_data, formula_data=None):
        """
        Args:
            xrd_data: (batch, seq_len) - XRD pattern
            formula_data: (batch, formula_dim) - Formula vector (optional)
        """
        # Extract XRD features
        x = xrd_data.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        xrd_features = torch.cat([avg_pool, max_pool], dim=1)
        xrd_features = xrd_features.view(xrd_features.size(0), -1)

        # Fuse formula features if provided
        if self.use_formula and formula_data is not None:
            formula_features = self.formula_encoder(formula_data)
            combined_features = torch.cat([xrd_features, formula_features], dim=1)
        else:
            combined_features = xrd_features

        # Classification
        output = self.classifier(combined_features)
        return output
