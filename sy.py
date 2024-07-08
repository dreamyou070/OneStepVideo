import torch
elem = torch.randn(16,16)
attnmaps = [elem, elem]
attnmaps = torch.stack(attnmaps, dim=0).mean(dim=0)
print(attnmaps.shape)