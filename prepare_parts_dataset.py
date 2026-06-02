#!/usr/bin/env python3
"""Convert the HITL Supervisely part layer into the project COCO dataset.

The source release has historically swapped folder names: ``Car damages
dataset`` contains the 998 part-annotated images. The conversion is explicit
here so a new download does not silently select the damage layer.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

PART_CLASSES = [
    "Back-bumper", "Back-door", "Back-wheel", "Back-window", "Back-windshield",
    "Fender", "Front-bumper", "Front-door", "Front-wheel", "Front-window",
    "Grille", "Headlight", "Hood", "License-plate", "Mirror", "Quarter-panel",
    "Rocker-panel", "Roof", "Tail-light", "Trunk", "Windshield",
]


def polygon_area(points):
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2.0


def polygon_box(points):
    xs, ys = zip(*points)
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def make_coco():
    return {
        "info": {"description": "SAM3 vehicle parts", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": index, "name": name}
            for index, name in enumerate(PART_CLASSES, start=1)
        ],
    }


def convert(source_root, output_root, validation_fraction=0.15, seed=13, copy_images=True):
    # This folder-name swap is present in the upstream HITL download.
    source_layer = source_root / "Car damages dataset"
    image_root = source_layer / "File1" / "img"
    annotation_root = source_layer / "File1" / "ann"
    if not image_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(
            "Expected Supervisely part layer at "
            f"{image_root} and {annotation_root}"
        )

    annotation_files = sorted(annotation_root.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No Supervisely annotations found in {annotation_root}")
    shuffled = list(range(len(annotation_files)))
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(len(annotation_files) * validation_fraction))
    validation_ids = set(shuffled[:validation_count])

    datasets = {"train": make_coco(), "val": make_coco()}
    next_image_id = {"train": 1, "val": 1}
    next_annotation_id = {"train": 1, "val": 1}

    for index, annotation_file in enumerate(annotation_files):
        split = "val" if index in validation_ids else "train"
        record = json.loads(annotation_file.read_text())
        image_name = annotation_file.name.removesuffix(".json")
        image_path = image_root / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by annotation is missing: {image_path}")

        image_id = next_image_id[split]
        next_image_id[split] += 1
        width = int(record.get("size", {}).get("width", 0))
        height = int(record.get("size", {}).get("height", 0))
        if not width or not height:
            from PIL import Image
            with Image.open(image_path) as image:
                width, height = image.size
        datasets[split]["images"].append({
            "id": image_id, "file_name": image_name,
            "width": width, "height": height,
        })

        for object_record in record.get("objects", []):
            class_name = object_record.get("classTitle")
            if class_name not in PART_CLASSES or object_record.get("geometryType") != "polygon":
                continue
            points = object_record.get("points", {}).get("exterior", [])
            if len(points) < 3:
                continue
            points = [[float(x), float(y)] for x, y in points]
            datasets[split]["annotations"].append({
                "id": next_annotation_id[split],
                "image_id": image_id,
                "category_id": PART_CLASSES.index(class_name) + 1,
                "segmentation": [[coordinate for point in points for coordinate in point]],
                "area": polygon_area(points),
                "bbox": polygon_box(points),
                "iscrowd": 0,
            })
            next_annotation_id[split] += 1

        if copy_images:
            destination = output_root / f"{split}_imgs" / image_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)

    output_root.mkdir(parents=True, exist_ok=True)
    for split, dataset in datasets.items():
        (output_root / f"{split}.json").write_text(json.dumps(dataset, indent=2))
        print(
            f"{split}: {len(dataset['images'])} images, "
            f"{len(dataset['annotations'])} boxes"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-copy-images", action="store_true")
    args = parser.parse_args()
    convert(
        args.source_root,
        args.output_root,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        copy_images=not args.no_copy_images,
    )


if __name__ == "__main__":
    main()
