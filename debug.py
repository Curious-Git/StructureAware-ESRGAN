from PIL import Image
lr = Image.open("test_images/test_lr.png")
hr = Image.open("test_images/test_hr.png")
print("LR size:", lr.size)
print("HR size:", hr.size)
