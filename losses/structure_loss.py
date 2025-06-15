import torch
import torch.nn.functional as F
import cv2
import numpy as np

class StructureAwareLoss(torch.nn.Module):
    def __init__(self, method='sobel'):
        super().__init__()
        self.method = method

    def extract_edges(self, img_tensor):
        img_np = img_tensor.detach().cpu().numpy()
        edge_maps = []
        for i in range(img_np.shape[0]):
            img = np.transpose(img_np[i], (1, 2, 0)) * 255.0
            img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            if self.method == 'sobel':
                edges = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)
            elif self.method == 'canny':
                edges = cv2.Canny(img_gray, 100, 200)
            edge_maps.append(edges / 255.0)
        edge_tensor = torch.tensor(np.expand_dims(np.array(edge_maps), 1), dtype=torch.float32).to(img_tensor.device)
        return edge_tensor

    def forward(self, sr_img, hr_img):
        sr_edges = self.extract_edges(sr_img)
        hr_edges = self.extract_edges(hr_img)
        return F.l1_loss(sr_edges, hr_edges)
