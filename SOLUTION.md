# Solution Report

## Reproducibility

Recommended environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If the standard torchvision downloads work in your environment, run:

```bash
.venv/bin/python validate.py --data_dir ./data --batch_size 64 --n_batches 128 --output results.json
```

On my machine the public dataset URL had a certificate issue, so I used a workspace-local Torch cache and pre-downloaded the official files:

```bash
mkdir -p data torch_cache/hub/checkpoints
curl -L -k https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -o data/cifar-100-python.tar.gz
tar -xzf data/cifar-100-python.tar.gz -C data
curl -L -k https://download.pytorch.org/models/resnet18-f37072fd.pth -o torch_cache/hub/checkpoints/resnet18-f37072fd.pth
TORCH_HOME=./torch_cache .venv/bin/python validate.py --data_dir ./data --batch_size 64 --n_batches 128 --output results.json
```

Final official run:

```json
{
  "val_accuracy_top1_imagenet_head": 0.0037,
  "val_accuracy_top1_init_head": 0.2492,
  "val_accuracy_top1_finetuned": 0.2726,
  "n_batches": 128,
  "batch_size": 64,
  "layers_tuned": ["fc.weight", "fc.bias"]
}
```

## Final Approach

The project fine-tunes a pretrained ResNet18 on CIFAR100 under a zero-order constraint: the optimizer cannot use backpropagation and can only derive updates from scalar loss evaluations.

I modified the allowed implementation files:

- `head_init.py`: initializes the 100-way CIFAR head from semantically related ImageNet classifier rows, averaging multiple rows when useful and preserving row norms.
- `zo_optimizer.py`: replaces per-parameter finite differences with SPSA-style simultaneous perturbations. It uses three central-difference directions per step, a short bias-only warmup, row-scaling directions for the classifier weights, momentum, update clipping, and rollback if the current mini-batch loss does not improve.
- `train_data.py`: uses a deterministic balanced subset with 120 samples per class.
- `augmentation.py`: adds moderate crop, color jitter, horizontal flip, and random erasing for training only.

The largest gain comes from the semantic head initialization: it raises the no-finetuning checkpoint to 24.92%. The zero-order optimizer then adds about 2.34 absolute accuracy points, reaching 27.26%.

## Discarded Ideas

- The skeleton coordinate-wise finite-difference estimator was discarded because it scales with the number of parameters and is too expensive for the final linear layer.
- Pure dense SPSA over all classifier weights was too noisy conceptually for the small budget, so the final version biases directions toward per-class row rescaling with a small dense component.
- Tuning deeper ResNet layers was discarded because the parameter count and black-box variance are much higher than for the classifier head.
- Stronger augmentation was avoided because repeated noisy transformed batches make zero-order loss comparisons less reliable.
