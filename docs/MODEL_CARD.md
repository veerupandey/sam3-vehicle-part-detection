# Model card

## Summary

This is a SAM 3.1-based, prompt-conditioned vehicle exterior part detector.
At inference time it runs one text prompt per supported part class and emits
COCO-style bounding boxes with confidence scores. The fine-tuning configuration
disables the segmentation head.

## Supported classes

The COCO annotations define 21 classes: Back-bumper, Back-door, Back-wheel,
Back-window, Back-windshield, Fender, Front-bumper, Front-door, Front-wheel,
Front-window, Grille, Headlight, Hood, License-plate, Mirror, Quarter-panel,
Rocker-panel, Roof, Tail-light, Trunk, and Windshield.

Read identifiers from the supplied COCO JSON. The IDs in an earlier delivery
document did not match the transferred annotations.

## Evaluation

Historical predictions were evaluated with standard COCO bounding-box metrics
on 149 validation images and 2,450 annotated instances.

| Metric | Value |
| --- | ---: |
| AP | 0.731 |
| AP@50 | 0.917 |
| AP@75 | 0.788 |
| AP small | 0.484 |
| AP medium | 0.727 |
| AP large | 0.806 |
| AR@100 | 0.808 |

Strict AP (IoU 0.50:0.95) was highest for Hood (0.899), Windshield (0.888),
Back-door (0.879), Front-door (0.879), and Front-wheel (0.861). It was lowest
for Roof (0.507), Mirror (0.547), Trunk (0.556), Quarter-panel (0.600), and
Tail-light (0.623).

The training and validation sets have no exact duplicate image content by
SHA-256 (849 unique train images, 149 unique validation images). This does not
establish independence between related photos, vehicle identities, or prior
training data.

## Training configuration

- Base model: SAM 3.1
- Optimizer: AdamW
- Epochs: 12
- Input: square 1008-pixel resize; mean/std `[0.5, 0.5, 0.5]`
- Training hardware: NVIDIA L40S, bfloat16
- Instance queries: 200
- Loss: Hungarian matching with box, GIoU, classification/presence, and
  one-to-many auxiliary matching terms

The original complete and resolved configurations are retained in the project
root. They preserve remote absolute paths and should be adapted before a new
training run. See [training details](TRAINING.md) for the data pipeline,
optimization schedule, loss construction, and evaluation protocol.

## Intended use and limitations

Use the model to localize common exterior passenger-vehicle parts in still
images. The result demonstrates that a SAM 3 image detector can be adapted to
a focused domain with 849 labeled training images. It does not determine
left/right laterality, assess damage, identify vehicle make/model, or produce
segmentation masks in this configuration.

Results may degrade for unusual vehicle types, occlusion, atypical views,
heavy damage, low resolution, reflections, small parts, and classes with lower
strict AP. Validate it on data representative of the intended operational
environment before deployment.

## Distribution

Do not assume the checkpoint or dataset may be redistributed with this
repository. Confirm their source licences, SAM 3 terms, and your organization's
rights before publishing or serving model outputs.
