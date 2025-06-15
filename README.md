# Structure-Aware ESRGAN

Enhanced ESRGAN model with Structure-Aware Perceptual Loss for improved edge and detail preservation in super-resolution.

## Features
- RRDB-based ESRGAN architecture
- Structure-aware loss using Sobel/Canny edge detectors
- Perceptual loss using pretrained VGG19
- Adversarial loss with discriminator
- DIV2K dataset support

## Setup

```bash
pip install -r requirements.txt
pip install git+https://github.com/xinntao/BasicSR.git@40b45fa
```

## Training

```bash
python train.py --config config/train_config.json
```


