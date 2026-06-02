"""Strip optimizer and resume state from a SAM training checkpoint."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Full training checkpoint")
    parser.add_argument("destination", type=Path, help="Inference-only output")
    args = parser.parse_args()

    checkpoint = torch.load(
        args.source, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a training checkpoint containing a model mapping")
    model = checkpoint["model"]
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model,
            "checkpoint_type": "inference_only",
            "source_epoch": checkpoint.get("epoch"),
        },
        args.destination,
    )
    model_bytes = sum(
        value.numel() * value.element_size()
        for value in model.values()
        if isinstance(value, torch.Tensor)
    )
    print(f"wrote {args.destination} ({args.destination.stat().st_size} bytes)")
    print(f"model tensor bytes: {model_bytes}")
    print("optimizer, scaler, and training-progress state were removed")


if __name__ == "__main__":
    main()
