import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.rrdb import RRDBNet
from losses.structure_loss import StructureAwareLoss
from utils.dataloader import SRDataset
import json, os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def train(config, resume_checkpoint=None):
    # Build model
    model = RRDBNet(
        in_channels=3,
        out_channels=3,
        num_feat=config.get("num_feat", 64),
        num_blocks=config.get("num_blocks", 23),
        growth_channels=config.get("growth_channels", 32)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    start_epoch = 0

    # 🔁 Resume logic
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Resumed training from epoch {start_epoch}")

    # Data
    dataset = SRDataset(config["train_lr_dir"], config["train_hr_dir"], config["image_size"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Loss
    criterion = nn.L1Loss()
    structure_loss = StructureAwareLoss().to(device)

    os.makedirs("results", exist_ok=True)

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        running_loss = 0.0
        for lr, hr in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)

            loss_pix = criterion(sr, hr)
            loss_struct = structure_loss(sr, hr)
            loss = loss_pix + 0.1 * loss_struct

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Save model every 2 epochs
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"results/model_epoch{epoch+1}.pth")

        # Save full checkpoint (for resuming)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, "results/latest_checkpoint.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_config.json")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_checkpoint=args.resume)
