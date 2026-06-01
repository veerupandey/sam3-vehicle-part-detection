# Training details

## Objective

The training task is prompt-conditioned detection of 21 exterior vehicle parts.
For each image and category prompt, the model predicts up to 200 candidate
instances. The saved configuration disables the segmentation head, so the
supervised output is bounding boxes rather than masks.

## Dataset

The training source is the Humans-in-the-Loop car-parts/damages dataset in
COCO format. The transferred annotation files contain the following:

| Split | Images | Box instances | Categories |
| --- | ---: | ---: | ---: |
| Train | 849 | 13,317 | 21 |
| Validation | 149 | 2,450 | 21 |

The categories cover common exterior body parts, glazing, lights, wheels, and
small components such as mirrors and license plates. The original annotations
also include segmentation information, but this run uses boxes only.

An exact SHA-256 audit found no duplicate image content between the two splits.
That is a narrow duplicate check; it does not rule out related photos of the
same vehicle, adjacent frames, or external pretraining exposure.

## Input pipeline

During training, the data pipeline:

1. removes crowd annotations and empty targets;
2. decodes available RLE annotations;
3. perturbs input boxes with standard deviation 0.1 and a maximum displacement
   of 20 pixels;
4. applies random resize/scale jitter from 480 to 1008 pixels, preserving a
   square 1008-pixel output by padding;
5. converts to tensors and normalizes each channel with mean and standard
   deviation `[0.5, 0.5, 0.5]`.

Validation uses a deterministic 1008-pixel square resize and the same 0.5
normalization, without the training-time jitter or box perturbation.

## Model and optimization

| Component | Configuration |
| --- | --- |
| Starting point | SAM 3.1 multiplex checkpoint |
| Visual backbone | Fine-tuned; layer-wise decay 0.9 |
| Text tower | Fine-tuned |
| Detection queries | 200 |
| Segmentation | Disabled |
| Optimizer | AdamW, weight decay 0.1 |
| Transformer learning rate | 8e-5 |
| Vision-backbone learning rate | 2.5e-5 |
| Language-backbone learning rate | 5e-6 |
| Learning-rate schedule | Inverse square root, 20-step warmup and cooldown |
| Batch size | 1 for training and validation |
| Epochs | 12 |
| Seed | 123 |
| Hardware | One NVIDIA L40S GPU, bfloat16 AMP |

The loss combines Hungarian matching with classification, box L1, generalized
IoU, and presence terms. Matching costs are classification 2.0, box 5.0, and
GIoU 2.0. The run also enables a one-to-many auxiliary matcher with top-k 4,
threshold 0.4, alpha 0.3, and weight 2.0.

The final `hitl_parts_v2` run resumed from an earlier `hitl_parts_full`
checkpoint and saved a checkpoint every two epochs. The exact paths are
preserved in `config_resolved.yaml` for provenance but must be changed for a
new environment.

## Evaluation protocol

The historical result uses COCO bounding-box evaluation with up to 100
detections per image. The validation pipeline includes negative categories and
chunks categories in groups of 20. The result reported in the README was
recomputed from the saved prediction JSON and the transferred `val.json`.

For a fresh evaluation, use `eval_per_class.py`. It reports both AP averaged
over IoU thresholds 0.50:0.95 and AP@50, avoiding the earlier script's
mistake of labeling AP@50 as multi-IoU AP.

## Local pre-fine-tuning reference benchmark

The configured starting point was `sam3.1_multiplex.pt`. That exact checkpoint
resides on the stopped Lightning studio and was not available locally when this
benchmark was run. A locally available upstream `sam3.pt` checkpoint was
architecture-compatible with the detection model: it loaded every detector
parameter, with only the disabled segmentation-head weights excluded.

The following is therefore a **SAM 3 baseline**, not an exact SAM 3.1
pre-fine-tuning reproduction. It uses the same deterministic eight validation
image IDs (`7, 27, 29, 36, 58, 63, 71, 140`), category prompts, box processing,
and MPS execution path as the fine-tuned smoke run.

| Checkpoint | AP | AP@50 | AP@75 |
| --- | ---: | ---: | ---: |
| Upstream SAM 3 baseline | 0.427 | 0.517 | 0.461 |
| Fine-tuned vehicle-parts checkpoint | 0.795 | 0.942 | 0.834 |
| Difference | +0.368 | +0.425 | +0.373 |

The baseline confirms a substantial improvement on the matched sample. Run the
exact `sam3.1_multiplex.pt` over all 149 validation images after the Lightning
studio is available to obtain the definitive before/after comparison.
