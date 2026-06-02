"""Evaluate COCO bounding-box predictions and report AP per category."""

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, help="COCO ground-truth JSON")
    parser.add_argument("--predictions", required=True, help="COCO predictions JSON")
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument("--image-ids", type=int, nargs="+")
    args = parser.parse_args()

    ground_truth = COCO(args.gt)
    predictions = json.loads(Path(args.predictions).read_text())
    evaluator = COCOeval(ground_truth, ground_truth.loadRes(predictions), "bbox")
    if args.image_ids:
        evaluator.params.imgIds = args.image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    result = {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "images": len(evaluator.params.imgIds),
        "per_class": {},
    }
    for index, category_id in enumerate(evaluator.params.catIds):
        category = ground_truth.cats[category_id]
        precision = evaluator.eval["precision"][:, :, index, 0, 2]
        precision_at_50 = precision[0]
        valid = precision[precision > -1]
        valid_at_50 = precision_at_50[precision_at_50 > -1]
        result["per_class"][category["name"]] = {
            "AP": float(valid.mean()) if len(valid) else None,
            "AP50": float(valid_at_50.mean()) if len(valid_at_50) else None,
        }
        print(category["name"], result["per_class"][category["name"]])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
