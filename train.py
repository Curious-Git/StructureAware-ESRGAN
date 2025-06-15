import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.rrdb import RRDB
from losses.structure_loss import StructureAwareLoss
from utils.dataloader import SRDataset
import json, os
from tqdm import tqdm
from models.rrdb import RRDBNet
generator=RRDBNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def train(config):
    model = RRDBNet(in_channels=3, out_channels=3, num_feat=64, num_blocks=23).to(device)
    dataset = SRDataset(config["train_lr_dir"], config["train_hr_dir"], config["image_size"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    criterion = nn.L1Loss()
    structure_loss = StructureAwareLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    os.makedirs("results", exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for lr, hr in tqdm(dataloader):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)

            loss_pix = criterion(sr, hr)
            loss_struct = structure_loss(sr, hr)
            loss = loss_pix + 0.1 * loss_struct

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {running_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), f"results/model_epoch{epoch+1}.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
