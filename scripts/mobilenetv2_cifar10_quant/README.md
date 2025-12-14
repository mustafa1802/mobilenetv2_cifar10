# Quantization Infrastructure

## Usage 
`pip install -r requirements.txt`

## Basic QAT training (4-bit weights, 8-bit activations, no W&B)
```
python mobilenetv2_cifar10_compression.py \
  --data-dir ./data \
  --epochs 15 \
  --batch-size 128 \
  --lr 0.005 \
  --weight-decay 4e-5 \
  --weight-bit-width 4 \
  --activation-bit-width 8
```
## QAT with W&B logging enabled

Make sure WANDB_API_KEY is set in your environment, then:
```
export WANDB_API_KEY=YOUR_KEY_HERE

python mobilenetv2_cifar10_compression.py \
  --data-dir ./data \
  --epochs 15 \
  --batch-size 128 \
  --lr 0.005 \
  --weight-decay 4e-5 \
  --weight-bit-width 4 \
  --activation-bit-width 8 \
  --wandb-project mobilenetv2_cifar10_qat \
  --use-wandb
```

## 8-bit / 8-bit QAT (more typical deployment setting)
export WANDB_API_KEY=YOUR_KEY_HERE
```
python mobilenetv2_cifar10_compression.py \
  --data-dir ./data \
  --epochs 15 \
  --batch-size 128 \
  --lr 0.005 \
  --weight-decay 4e-5 \
  --weight-bit-width 4 \
  --activation-bit-width 8 \
  --wandb-project mobilenetv2_cifar10_qat \
  --use-wandb
```

## Leave first & last layers in FP32
```
python mobilenetv2_cifar10_compression.py \
  --data-dir ./data \
  --epochs 15 \
  --batch-size 128 \
  --weight-bit-width 4 \
  --activation-bit-width 8 \
  --no-quantize-first-last
```

## Start from an FP32 checkpoint
```
python mobilenetv2_cifar10_compression.py \
  --data-dir ./data \
  --epochs 10 \
  --batch-size 128 \
  --weight-bit-width 4 \
  --activation-bit-width 8 \
  --fp32-checkpoint path/to/fp32_mobilenetv2_cifar10.pth
```

## Run as a W&B sweep agent
First create a sweep (this script does that for you) and run multiple QAT configs:
```
export WANDB_API_KEY=YOUR_KEY_HERE

python mobilenetv2_cifar10_compression.py \
  --mode sweep \
  --use-wandb \
  --wandb-project mobilenetv2_cifar10_qat \
  --sweep-count 16
```
