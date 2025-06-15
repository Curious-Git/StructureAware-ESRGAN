# Structure-Aware ESRGAN

This project enhances ESRGAN with a custom **Structure-Aware Perceptual Loss** that better preserves edges and structural details in super-resolved images.

## 🔧 Features
- Integrated Sobel/Canny edge loss module
- Combines Perceptual + Adversarial + Structure losses
- Compatible with DIV2K, Set5/14 datasets

## 🧠 Loss Function

Total Loss = perceptual_loss + adversarial_loss + structure_loss

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/StructureAware-ESRGAN.git
cd StructureAware-ESRGAN
pip install -r requirements.txt
```

## 🏃 Train

```bash
python train.py --config config/train_config.json
```

## 📈 Evaluate

```bash
python test.py --input data/val
```
