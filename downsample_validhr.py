from PIL import Image
img = Image.open("data/DIV2K_valid_HR/0801.png")
lr_img = img.resize((img.width // 4, img.height // 4), Image.BICUBIC)
lr_img.save("test_images/test_lr.png")     # Save downscaled input
img.save("test_images/test_hr.png")        # Save ground truth HR
