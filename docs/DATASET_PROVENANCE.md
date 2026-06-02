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

The project COCO files identify themselves as `SAM3 vehicle parts`, version
`1.0`, and do not retain licence metadata in their `licenses` field. The
original Kaggle source remains the licence authority.

## Data preparation and split

The original Lightning training script was recovered from the remote artifact
directory. It confirms an upstream folder-name swap: `Car damages dataset`
contains the 998 part-annotated images, while `Car parts dataset` contains the
814 damage-annotated images. The script selected the former, sorted its
Supervisely annotation filenames, shuffled their indices with
`random.Random(13)`, and assigned the first `int(998 * 0.15) = 149` records to
validation. The remaining 849 records became training data. This is a random,
non-stratified split.

`prepare_parts_dataset.py` is the cleaned, standalone implementation of that
conversion. It converts Supervisely polygons to COCO instances while retaining
the 21 part classes and can copy the corresponding images. Its default seed and
validation fraction match the recovered training script. Exact filenames still
depend on the downloaded Kaggle release, so verify the resulting counts before
training.

An SHA-256 content audit of the delivered files found no exact image duplicate
between train and validation. It does not rule out related images of the same
vehicle or adjacent captures.

## Download guidance

Download the source from its Kaggle page using an account and method permitted
by Kaggle’s terms. Keep the original source material separate from this
repository; `data/` is ignored by Git. To recreate the derived dataset:

```sh
python prepare_parts_dataset.py \
  --source-root /path/to/car-parts-and-car-damages \
  --output-root data/parts
```
