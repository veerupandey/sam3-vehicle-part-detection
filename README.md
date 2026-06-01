# SAM 3.1 Vehicle Part Detection

Fine-tuned SAM 3.1 detector for 21 exterior vehicle-part classes. The model
produces bounding boxes from a vehicle image and a text prompt for each part
class. It was trained with segmentation disabled, so this repository evaluates
**detection boxes**, not masks.

## How it was trained

The detector was fine-tuned from the SAM 3.1 multiplex checkpoint on a COCO
vehicle-parts dataset with 21 exterior-part classes. The transferred dataset
contains 849 training images with 13,317 instances and 149 validation images
with 2,450 instances.

Training used 12 epochs on one NVIDIA L40S GPU with bfloat16 mixed precision,
AdamW, 200 instance queries, and a batch size of one. Images were resized to a
1008-pixel square; training used scale jitter from 480 to 1008 pixels plus
bounded box-noise augmentation. The image and text towers were both
fine-tuned. See [training details](docs/TRAINING.md).

## Results

The supplied historical prediction file was re-evaluated against the current
COCO validation annotations (149 images):

| Metric | Score |
| --- | ---: |
| COCO AP (IoU 0.50:0.95) | 0.731 |
| AP@50 | 0.917 |
| AP@75 | 0.788 |
| AR@100 | 0.808 |

AP@50 is average precision at an IoU threshold of 0.50; it is not a simple
percentage of correctly classified images. The supplied prediction file was
re-evaluated against the current validation annotations to produce these values.

See [the model card](docs/MODEL_CARD.md) for per-class results, data details,
limitations, and intended use.

## Repository layout

```text
.
├── config.yaml                 # Original training configuration
├── config_resolved.yaml        # Fully resolved original configuration
├── run_mps.py                  # Apple Silicon MPS validation runner
├── eval_per_class.py           # COCO box evaluator with per-class AP/AP@50
├── vendor/sam3/                # SAM 3 source adapted for image-only MPS use
├── docs/                       # Model card, training, and compatibility notes
└── results/                    # Local evaluation outputs; ignored by Git
```

Model weights, datasets, downloaded remote artifacts, and local Python
environments are intentionally excluded from Git. See `.gitignore`.

## Local evaluation

An Apple Silicon MPS runner is included for local evaluation only. It was
verified with Python 3.12, PyTorch 2.6.0, and torchvision 0.21.0. The original
training and historical evaluation used CUDA and bfloat16.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-mps.txt
```

Install a PyTorch/torchvision pair appropriate for your platform first if the
requirements resolver does not select one automatically.

## Data and checkpoint setup

The repository does not distribute either the dataset or the 10,009,567,404
byte checkpoint. Obtain them under the licences and permissions that apply to
your organization.

Expected paths:

```text
data/parts/
├── val.json
└── val_imgs/
    └── <vehicle images>

checkpoints/
└── checkpoint.pt
```

The verified checkpoint SHA-256 is:

```text
89033c6d9f7a3851a0d8c36621f48cbe98dd73f6d31ae97d5557637a9278e54a
```

Verify it after download:

```sh
shasum -a 256 checkpoints/checkpoint.pt
```

### Run MPS evaluation

Run a deterministic eight-image validation smoke evaluation:

```sh
.venv/bin/python run_mps.py \
  --checkpoint checkpoints/checkpoint.pt \
  --data-root data/parts \
  --limit 8 \
  --output results/mps-smoke
```

Use `--limit 0` for every validation image. The runner writes COCO-format
predictions, aggregate metrics, timing data, metadata, and side-by-side ground
truth/prediction visualizations to the output directory.

## Re-evaluate existing predictions

```sh
.venv/bin/python eval_per_class.py \
  --gt data/parts/val.json \
  --predictions predictions.json \
  --output results/evaluation.json
```

## Notes for maintainers

- Use category IDs from `val.json`; do not hard-code a separate taxonomy.
- The actual transferred annotations contain 849 training images and 149
  validation images. The earlier delivery README's 1,098/122 split is stale.
- Original training preprocessing uses normalization mean/std `[0.5, 0.5,
  0.5]` and square 1008-pixel resize. It does not use ImageNet normalization.
- MPS runs in float32 and uses compatibility substitutions for a few
  CUDA-only operations. See [MPS compatibility](docs/MPS_COMPATIBILITY.md).
- Before publishing, review the applicable SAM 3 license, the dataset license,
  and any restrictions on distributing fine-tuned weights.
