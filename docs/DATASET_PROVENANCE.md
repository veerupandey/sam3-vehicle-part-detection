# Dataset provenance

## Source dataset

The vehicle-part images originate from [Car Parts and Car Damages on Kaggle](https://www.kaggle.com/datasets/humansintheloop/car-parts-and-car-damages), published by Humans in the Loop. The transferred annotation records preserve the original local download path:

```text
kagglehub/datasets/humansintheloop/car-parts-and-car-damages/versions/2/
```

This indicates the source was downloaded through KaggleHub at dataset version
2. Kaggle’s dataset card describes 1,812 total images: 998 with car-part
polygons and 814 with car-damage polygons. It states the source dataset is
dedicated to the public domain under CC0 1.0.

## What this project used

This project uses the 998 car-part images, not the 814 car-damage images. The
received project COCO annotations contain exactly 998 images in total:

| Split | Images | Box instances |
| --- | ---: | ---: |
| Train | 849 | 13,317 |
| Validation | 149 | 2,450 |

The source annotations are polygons. The project dataset is a derived COCO
representation for vehicle-part **box detection**: it retains the 21 part
classes and supplies boxes used by the SAM 3 fine-tuning run. Segmentation is
disabled in the saved training configuration, so source polygons were not used
as mask supervision in this run.

The project COCO files identify themselves as `SAM3 vehicle parts`, version `1.0`, and
do not retain licence metadata in their `licenses` field. The original Kaggle
source remains the licence authority.

## Data preparation boundary

The final train/validation COCO files and image folders were already present
on the Lightning training host. The records establish the source dataset,
version, derived annotation format, class taxonomy, and final split sizes, but
the script that produced the 849/149 split was not included in the delivered
artifacts. This repository therefore treats that split as a recorded training
artifact rather than claiming it can recreate the exact partition from only the
source download.

An SHA-256 content audit of the delivered files found no exact image duplicate
between train and validation. It does not rule out related images of the same
vehicle or adjacent captures.

## Download guidance

Download the source from its Kaggle page using an account and method permitted
by Kaggle’s terms. Keep the original source material separate from this
repository; `data/` is ignored by Git. After downloading, reproduce the
project’s COCO derivative and split only if the missing conversion/split script
is recovered or replaced with a documented new procedure.
