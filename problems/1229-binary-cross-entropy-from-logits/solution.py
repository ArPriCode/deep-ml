import torch

def bce_with_logits(logits, targets):
    loss = torch.clamp(logits, min=0) - logits * targets + torch.log(1 + torch.exp(-torch.abs(logits)))
    return round(torch.mean(loss).item(), 4)