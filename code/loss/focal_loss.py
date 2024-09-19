import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Convert inputs to probabilities
        logpt = nn.functional.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)

        # Gather the log-probabilities at the indices of the target class
        logpt = logpt.gather(1, targets.unsqueeze(1))
        logpt = logpt.view(-1)

        pt = pt.gather(1, targets.unsqueeze(1))
        pt = pt.view(-1)

        # Apply the weights
        if self.alpha is not None:
            at = self.alpha.gather(0, targets.view(-1))
            logpt = logpt * at

        # Apply the focal loss modulating factor
        loss = -1 * (1 - pt) ** self.gamma * logpt

        # Mask out the loss for the ignored index (class 0)
        loss = loss * (targets.view(-1) != self.ignore_index).float()

        return loss.mean()
