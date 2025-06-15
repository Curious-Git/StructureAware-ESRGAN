
import os
import cv2

hr_dir = "data/DIV2K_train_HR"
lr_dir = "data/train_LR"
scale = 4  # 4x downscaling

os.makedirs(lr_dir, exist_ok=True)

for img_name in os.listdir(hr_dir):
    hr_path = os.path.join(hr_dir, img_name)
    lr_path = os.path.join(lr_dir, img_name)

    img = cv2.imread(hr_path)
    if img is None:
        print(f"⚠️ Skipping corrupted image: {img_name}")
        continue

    h, w = img.shape[:2]
    lr_img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(lr_path, lr_img)

print("✅ Done generating LR images.")
