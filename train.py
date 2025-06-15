from losses.structure_loss import StructureAwareLoss
import torch

# Dummy example
structure_loss = StructureAwareLoss(method='sobel')
sr_img = torch.rand(1, 3, 128, 128)
hr_img = torch.rand(1, 3, 128, 128)

loss = structure_loss(sr_img, hr_img)
print(f"Structure loss: {loss.item():.4f}")
