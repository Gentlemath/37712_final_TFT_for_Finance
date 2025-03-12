
import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            loss = torch.max(q * errors, (q - 1) * errors).mean()
            losses.append(loss)
        return torch.stack(losses).mean()  # Average over quantiles
    

if __name__ == "__main__":
    # Example usage
    quantiles = [0.1, 0.5, 0.9]
    criterion = QuantileLoss(quantiles)

    y_true = torch.tensor([10.0, 20.0, 30.0])
    y_pred = torch.tensor([[9.0, 10.0, 11.0], [19.0, 20.0, 21.0], [29.0, 30.0, 31.0]])  # Each row has quantile predictions

    loss = criterion(y_pred, y_true)
    print(loss.item())