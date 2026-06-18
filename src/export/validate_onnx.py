"""Validate PyTorch and ONNX token-classification predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoModelForTokenClassification, AutoTokenizer


def compare_logits(pytorch_logits: np.ndarray, onnx_logits: np.ndarray) -> dict[str, float | bool]:
    """Compare two logits arrays with a small numerical tolerance."""

    abs_diff = np.abs(pytorch_logits - onnx_logits)
    return {
        "same_shape": pytorch_logits.shape == onnx_logits.shape,
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "same_argmax": bool(np.array_equal(np.argmax(pytorch_logits, axis=-1), np.argmax(onnx_logits, axis=-1))),
    }


def validate(model_dir: str | Path, onnx_dir: str | Path, text: str) -> dict[str, float | bool]:
    """Run one sample through PyTorch and ONNX models and compare logits."""

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    pytorch_model = AutoModelForTokenClassification.from_pretrained(model_dir)
    onnx_model = ORTModelForTokenClassification.from_pretrained(onnx_dir)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    pytorch_outputs = pytorch_model(**inputs)
    onnx_outputs = onnx_model(**inputs)

    return compare_logits(
        pytorch_outputs.logits.detach().cpu().numpy(),
        onnx_outputs.logits.detach().cpu().numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument(
        "--text",
        default="Dibutuhkan kandidat yang menguasai Python, SQL, dan kemampuan komunikasi yang baik.",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    result = validate(args.model_dir, args.onnx_dir, args.text)
    print(json.dumps(result, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
