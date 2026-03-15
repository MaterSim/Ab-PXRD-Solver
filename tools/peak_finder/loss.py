import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # pred: (batch, 2), raw model scores
        # target: (batch,), ground truth labels (0 or 1)

        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * focal_loss
        return focal_loss.mean()
    

class PositionWeightedLoss(nn.Module):
    """Loss with position-based weighting (earlier peaks are more important)"""
    
    def __init__(self, xrd_length=3500, decay_rate=0.001, base_loss='focal'):
        super().__init__()
        self.xrd_length = xrd_length
        
        # Position weights: higher for earlier positions
        positions = torch.arange(xrd_length).float()
        weights = torch.exp(-decay_rate * positions)
        # Normalize to [0.5, 1.0]
        weights = 0.5 + 0.5 * (weights - weights.min()) / (weights.max() - weights.min())
        self.register_buffer('position_weights', weights)
        
        if base_loss == 'focal':
            self.base_loss = FocalLoss()
        else:
            self.base_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pred, target, positions):
        # pred: (batch, 2)
        # target: (batch,)
        # positions: (batch,) - center position for each sample
        
        if isinstance(self.base_loss, FocalLoss):
            loss = self.base_loss(pred, target)
            # Recalculate batch losses for weighting
            batch_losses = F.cross_entropy(pred, target, reduction='none')
            p_t = torch.exp(-batch_losses)
            focal_weights = (1 - p_t) ** 2.0
            batch_losses = focal_weights * batch_losses
        else:
            batch_losses = self.base_loss(pred, target)
        
        # Apply position weights
        pos_weights = self.position_weights[positions.long()]
        weighted_loss = batch_losses * pos_weights
        
        return weighted_loss.mean()
