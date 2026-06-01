"""Benchmark SAM 3 checkpoints for vehicle-part box detection on Apple MPS.

This is a detection benchmark. The fine-tuned checkpoint has segmentation
disabled, so it produces boxes and scores rather than pixel masks.
"""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "vendor"))

from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a SAM checkpoint")
    parser.add_argument(
        "--checkpoint-label",
        default="unspecified",
        help="Human-readable label saved in metadata.json",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Directory containing val.json and val_imgs/",
    )
    parser.add_argument(
        "--allow-extra-weights",
        action="store_true",
        help="Permit unused weights for disabled heads, such as segmentation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Number of deterministic validation images to benchmark; 0 uses all.",
    )
    parser.add_argument(
        "--output",
        default="results/local-benchmark",
        help="Directory for predictions, metrics, metadata, and visualizations.",
    )
    return parser.parse_args()


def configure_mps_rotary_embeddings(model):
    """Use upstream real-valued RoPE tensors because MPS lacks complex kernels."""
    for module in model.modules():
        frequencies = getattr(module, "freqs_cis", None)
        if not isinstance(frequencies, torch.Tensor) or not frequencies.is_complex():
            continue
        module.use_rope_real = True
        module.register_buffer(
            "freqs_cis_real", frequencies.real.contiguous(), persistent=False
        )
        module.register_buffer(
            "freqs_cis_imag", frequencies.imag.contiguous(), persistent=False
        )
        module._buffers.pop("freqs_cis", None)
        module.freqs_cis = frequencies


def load_model(args):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable in this PyTorch installation.")

    model = build_sam3_image_model(
        device="cpu", load_from_HF=False, enable_segmentation=False
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    state_dict = checkpoint.get("model", checkpoint)
    if any(key.startswith("detector.") for key in state_dict):
        state_dict = {
            key.removeprefix("detector."): value
            for key, value in state_dict.items()
            if key.startswith("detector.")
        }

    if args.allow_extra_weights:
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            raise RuntimeError(
                "Checkpoint does not contain every detector weight: "
                f"{load_result.missing_keys}"
            )
    else:
        load_result = model.load_state_dict(state_dict, strict=True)

    metadata = {
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_top_level_keys": list(checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": "mps",
        "dtype": "float32",
        "load_result": str(load_result),
        "resolution": 1008,
        "task": "bounding-box detection; segmentation disabled",
        "torch": torch.__version__,
    }
    del state_dict, checkpoint
    gc.collect()

    configure_mps_rotary_embeddings(model)
    return model.to("mps").eval(), metadata


def select_images(dataset, limit):
    images = sorted(dataset["images"], key=lambda image: image["id"])
    if limit < 0:
        raise ValueError("--limit must be zero or a positive integer")
    if limit:
        images = sorted(
            random.Random(42).sample(images, min(limit, len(images))),
            key=lambda image: image["id"],
        )
    return images


def draw_comparison(image, dataset, image_id, predictions, destination):
    """Save ground truth and score-filtered detections side by side."""
    canvas = Image.new("RGB", (image.width * 2, image.height + 30), "white")
    canvas.paste(image, (0, 30))
    canvas.paste(image, (image.width, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), "Ground truth", fill="black")
    draw.text((image.width + 5, 5), "Predictions (score >= 0.35)", fill="black")
    category_names = {category["id"]: category["name"] for category in dataset["categories"]}

    for records, offset in ((dataset["annotations"], 0), (predictions, image.width)):
        for record in records:
            if record["image_id"] != image_id or record.get("score", 1.0) < 0.35:
                continue
            x, y, width, height = record["bbox"]
            color = f"hsl({record['category_id'] * 137 % 360},85%,45%)"
            draw.rectangle(
                (x + offset, y + 30, x + width + offset, y + height + 30),
                outline=color,
                width=2,
            )
            label = category_names[record["category_id"]]
            if "score" in record:
                label = f"{label} {record['score']:.2f}"
            draw.text(
                (x + offset, y + 30), label, fill=color, stroke_width=1, stroke_fill="white"
            )
    canvas.save(destination)


def evaluate_predictions(data_root, predictions, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO(str(data_root / "val.json"))
    evaluator = COCOeval(ground_truth, ground_truth.loadRes(predictions), "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "images": len(image_ids),
        "per_class": {},
    }
    for index, category in enumerate(ground_truth.loadCats(evaluator.params.catIds)):
        precision = evaluator.eval["precision"][:, :, index, 0, 2]
        precision_at_50 = precision[0]
        valid_precision = precision[precision > -1]
        valid_precision_at_50 = precision_at_50[precision_at_50 > -1]
        metrics["per_class"][category["name"]] = {
            "AP": float(valid_precision.mean()) if len(valid_precision) else None,
            "AP50": (
                float(valid_precision_at_50.mean()) if len(valid_precision_at_50) else None
            ),
        }
    return metrics


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output = Path(args.output)
    if not (data_root / "val.json").is_file() or not (data_root / "val_imgs").is_dir():
        raise FileNotFoundError("--data-root must contain val.json and val_imgs/")
    output.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    print("Building detector and loading checkpoint", flush=True)
    model, metadata = load_model(args)
    processor = Sam3Processor(model, device="mps", confidence_threshold=0.0)
    dataset = json.loads((data_root / "val.json").read_text())
    images = select_images(dataset, args.limit)
    metadata["model_load_seconds"] = time.perf_counter() - started_at
    print(f"Model ready in {metadata['model_load_seconds']:.2f}s", flush=True)

    predictions, timings = [], []
    with torch.inference_mode():
        for image_record in images:
            image_started_at = time.perf_counter()
            image_path = data_root / "val_imgs" / Path(image_record["file_name"]).name
            image = Image.open(image_path).convert("RGB")
            state = processor.set_image(image)
            torch.mps.synchronize()
            encoder_seconds = time.perf_counter() - image_started_at

            image_features = {
                key: state["backbone_out"][key]
                for key in ("backbone_fpn", "vision_features", "vision_pos_enc")
            }
            for category in dataset["categories"]:
                # forward_text mutates the feature dictionary in this checkpoint.
                text_features = model.backbone.forward_text([category["name"]], device="mps")
                state["backbone_out"].update(image_features)
                state["backbone_out"].update(text_features)
                raw_output = model.forward_grounding(
                    backbone_out=state["backbone_out"],
                    find_input=processor.find_stage,
                    geometric_prompt=model._get_dummy_prompt(),
                    find_target=None,
                )
                scores = (
                    raw_output["pred_logits"].sigmoid()
                    * raw_output["presence_logit_dec"].sigmoid().unsqueeze(1)
                ).flatten().cpu()
                boxes = box_cxcywh_to_xyxy(raw_output["pred_boxes"]).reshape(-1, 4).cpu()
                boxes *= torch.tensor([image.width, image.height, image.width, image.height])
                for box, score in zip(boxes.tolist(), scores.tolist()):
                    x1, y1, x2, y2 = box
                    predictions.append(
                        {
                            "image_id": image_record["id"],
                            "category_id": category["id"],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score,
                        }
                    )

            torch.mps.synchronize()
            elapsed_seconds = time.perf_counter() - image_started_at
            timings.append(
                {
                    "image_id": image_record["id"],
                    "encoder_seconds": encoder_seconds,
                    "total_seconds": elapsed_seconds,
                }
            )
            print(
                f"Image {image_record['id']}: {elapsed_seconds:.2f}s "
                f"({encoder_seconds:.2f}s encoder)",
                flush=True,
            )
            draw_comparison(
                image,
                dataset,
                image_record["id"],
                predictions,
                output / f"image_{image_record['id']:04d}.jpg",
            )
            del state, raw_output
            torch.mps.empty_cache()

    metadata["image_ids"] = [image["id"] for image in images]
    metadata["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
    metrics = evaluate_predictions(data_root, predictions, metadata["image_ids"])
    (output / "predictions.json").write_text(json.dumps(predictions))
    (output / "timings.json").write_text(json.dumps(timings, indent=2))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
