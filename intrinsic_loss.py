import torch.nn as nn

class IntrinsicLoss(nn.Module):
    def __init__(self, lambda_val=0.01):
        super(IntrinsicLoss, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, output, targets):
        # Compute the original loss
        original_loss = -torch.mean(torch.log(output))

        # Compute the intrinsic reward term
        novelity_scores = (1 - torch.softmax(-output, dim=1)).mean(dim=1)  # Note: rn this is a hack to compute novelty scores
        intrinsic_reward = self.lambda_val * (-novelty_scores).sum()

        # Return the combined loss
        return original_loss + intrinsic_reward

