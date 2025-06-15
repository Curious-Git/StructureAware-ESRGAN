from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, image_size):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.filenames = os.listdir(hr_dir)
        self.lr_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        self.hr_transform = transforms.Compose([
            transforms.Resize((image_size * 4, image_size * 4)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.hr_dir, self.filenames[idx])).convert("RGB")
        lr_img = Image.open(os.path.join(self.lr_dir, self.filenames[idx])).convert("RGB")
        return self.lr_transform(lr_img), self.hr_transform(hr_img)
