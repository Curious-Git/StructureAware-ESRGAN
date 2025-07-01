import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.rrdb import RRDBNet
from losses.structure_loss import StructureAwareLoss
from utils.dataloader import SRDataset
import json, os
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def train(config, resume_checkpoint=None):
    model = RRDBNet(
        in_channels=3,
        out_channels=3,
        num_feat=config.get("num_feat", 64),
        num_blocks=config.get("num_blocks", 23),
        growth_channels=config.get("growth_channels", 32)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    start_epoch = 0

    # Auto resume logic
    resume_path = resume_checkpoint or "results/model_epoch98.pth"
    if os.path.exists(resume_path):
        print(f"🔄 Found model weights. Resuming from: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device))
        start_epoch = 98 + 1  # Start from next epoch
        print(f"✅ Resumed training from epoch {start_epoch}")
    else:
        print("🆕 No checkpoint found. Starting fresh from epoch 0")


    dataset = SRDataset(config["train_lr_dir"], config["train_hr_dir"], config["image_size"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    criterion = nn.L1Loss()
    structure_loss = StructureAwareLoss().to(device)

    os.makedirs("results", exist_ok=True)
    os.makedirs("benchmarks", exist_ok=True)

    psnr_list, ssim_list, loss_list = [], [], []

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        running_loss = 0.0

        for lr, hr in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            sr = torch.clamp(sr, 0.0, 1.0)

            loss_pix = criterion(sr, hr)
            loss_struct = structure_loss(sr, hr)
            loss = loss_pix + 0.03 * loss_struct  # 👈 Adjusted weight for structure loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_list.append(avg_loss)

        # ✅ Evaluation over 5 random batches
        model.eval()
        psnr_sum = 0.0
        ssim_sum = 0.0
        eval_samples = 5
        with torch.no_grad():
            sample_indices = random.sample(range(len(dataloader)), min(eval_samples, len(dataloader)))
            for i, (val_lr, val_hr) in enumerate(dataloader):
                if i not in sample_indices:
                    continue
                val_lr, val_hr = val_lr.to(device), val_hr.to(device)
                val_sr = model(val_lr)
                val_sr = torch.clamp(val_sr, 0.0, 1.0)

                for j in range(val_sr.size(0)):
                    sr_np = val_sr[j].cpu().permute(1, 2, 0).numpy()
                    hr_np = val_hr[j].cpu().permute(1, 2, 0).numpy()

                    sr_np = (sr_np * 255).astype(np.uint8)
                    hr_np = (hr_np * 255).astype(np.uint8)

                    psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=255)
                    ssim = structural_similarity(hr_np, sr_np, channel_axis=2)

                    psnr_sum += psnr
                    ssim_sum += ssim

        num_eval_images = len(sample_indices) * val_sr.size(0)
        avg_psnr = psnr_sum / num_eval_images
        avg_ssim = ssim_sum / num_eval_images
        psnr_list.append(avg_psnr)
        ssim_list.append(avg_ssim)

        print(f"✅ Epoch {epoch+1}: Loss = {avg_loss:.4f} | PSNR = {avg_psnr:.2f} dB | SSIM = {avg_ssim:.4f}")

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"results/model_epoch{epoch+1}.pth")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, "results/latest_checkpoint.pth")

        np.save("benchmarks/psnr.npy", np.array(psnr_list))
        np.save("benchmarks/ssim.npy", np.array(ssim_list))
        np.save("benchmarks/loss.npy", np.array(loss_list))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_config.json")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_checkpoint=args.resume)
