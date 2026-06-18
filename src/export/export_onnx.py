"""Export a fine-tuned HuggingFace token classifier to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer


def export_to_onnx(model_dir: str | Path, output_dir: str | Path) -> None:
    """Export a token classification model to ONNX format."""

    source = Path(model_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    model = ORTModelForTokenClassification.from_pretrained(source, export=True)
    tokenizer = AutoTokenizer.from_pretrained(source)

    model.save_pretrained(target)
    tokenizer.save_pretrained(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to the best HuggingFace model directory")
    parser.add_argument("--output-dir", required=True, help="Directory where ONNX assets will be written")
    args = parser.parse_args()

    export_to_onnx(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
