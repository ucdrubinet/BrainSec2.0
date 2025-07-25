'''script to see model layers'''

import torch
import torch.nn.utils.prune as prune

seg_model = torch.load("/cache/plaquebox-paper/models/ResNet18_19.pkl")
print(seg_model)



