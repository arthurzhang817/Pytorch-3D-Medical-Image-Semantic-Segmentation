import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

def cross_entropy3d(input, target, weight=None, reduction="mean"):
    _, c, h, w, z = input.size()
    _, ht, wt, zt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.permute(0, 2, 3, 4, 1).contiguous().view(-1, c) #input size 
    target = target.view(-1) # target size

    loss = F.cross_entropy(
        input, target, weight=weight, reduction=reduction, ignore_index=250
    )
    return loss