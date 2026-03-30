import torch
import numpy as np

def state_loss(phi_pred, phi):
    return 0.5 * torch.sum((phi - phi_pred) ** 2, dim=-1).mean()

def prediction_variance(preds):
    assert isinstance(preds, list)
    assert isinstance(preds[0], torch.Tensor)
    preds = torch.stack(preds).cpu().numpy()
    vars_ = np.var(preds, axis=0)
    return np.mean(vars_, axis=-1)