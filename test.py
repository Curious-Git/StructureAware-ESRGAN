import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from models.rrdb import RRDBNet
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = RRDBNet(in_channels=3, out_channels=3, num_feat=64, num_blocks=23).to(device)
model.load_state_dict(torch.load("results/model_epoch5.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.ToTensor()

# Load LR image
lr_path = "test_images/test_lr.png"
lr_image = Image.open(lr_path).convert("RGB")
lr_tensor = transform(lr_image).unsqueeze(0).to(device)

# Super-resolve
with torch.no_grad():
    sr_tensor = model(lr_tensor)

# Clamp to [0,1] and save
sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0, 1)
output_dir = "test_images"
os.makedirs(output_dir, exist_ok=True)
save_image(sr_tensor, os.path.join(output_dir, "sr_output_fixed.png"))

# Evaluate if HR is available
hr_path = "test_images/test_hr.png"
if os.path.exists(hr_path):
    hr_image = Image.open(hr_path).convert("RGB")
    sr_image = Image.open(os.path.join(output_dir, "sr_output_fixed.png")).convert("RGB")

    # Match dimensions
    hr_np = np.array(hr_image)
    sr_np = np.array(sr_image)
    h, w = min(hr_np.shape[0], sr_np.shape[0]), min(hr_np.shape[1], sr_np.shape[1])
    hr_np = hr_np[:h, :w]
    sr_np = sr_np[:h, :w]

    # PSNR & SSIM
    psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=255)
    ssim = structural_similarity(hr_np, sr_np, win_size=7, channel_axis=-1)

    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
else:
    print("HR image not found for comparison.")
