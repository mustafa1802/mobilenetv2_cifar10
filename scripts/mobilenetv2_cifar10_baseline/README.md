# Baseline Training Script

## Pre-requisites 
```
pip install -r requirements.txt
```
## Train MobileNetV2 on CIFAR-10 from scratch (with ImageNet pretrained weights)
```
python mobilenetv2_train.py
```

## Train with Custom Hyperparameters
Example: 50 epochs, batch size 256, learning rate 0.01
```
python mobilenetv2_train.py --mode train \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.01
```

## Train without pretrained weights
```
python mobilenetv2_train.py --mode train --no-pretrained
```

## Resume Training From a Checkpoint
```
python mobilenetv2_train.py --mode train \
    --resume mobilenetv2_cifar10_best.pth
```

## Evaluation Only (no training)
Evaluate a saved model on the CIFAR-10 test set
```
python mobilenetv2_train.py --mode eval \
    --resume mobilenetv2_cifar10_best.pth
```
If no --resume is provided, it will try to load the default: `mobilenetv2_cifar10_best.pth`

## Full Combined Example
```
python mobilenetv2_train.py --mode train \
    --epochs 100 \
    --mixed-precision \
    --save-path checkpoints/mnv2_cifar10.pth \
    --resume checkpoints/mnv2_cifar10.pth
```
