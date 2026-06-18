"""Data utilities for SkillScope ID token classification."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split


LABEL_LIST = [
    "O",
    "B-HSkill",
    "I-HSkill",
    "B-SSkill",
    "I-SSkill",
    "B-Tech",
    "I-Tech",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def parse_conll(path: str | Path) -> list[dict[str, list[str]]]:
    """Parse a CoNLL/BIO file into HuggingFace-friendly sentence records."""

    file_path = Path(path)
    sentences: list[dict[str, list[str]]] = []
    tokens: list[str] = []
    tags: list[str] = []
    previous_sentence_id: str | None = None

    def flush_sentence() -> None:
        nonlocal tokens, tags
        if tokens:
            sentences.append({"tokens": tokens, "ner_tags": tags})
            tokens, tags = [], []

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                flush_sentence()
                previous_sentence_id = None
                continue

            parts = line.split("\t") if "\t" in line else line.split()
            lowered = [part.lower() for part in parts]
            if any(name in lowered for name in ["sentence", "sentence#", "word", "token", "tag"]):
                continue

            if len(parts) < 2:
                raise ValueError(f"Invalid CoNLL row at line {line_number}: {raw_line!r}")

            if len(parts) >= 4 and parts[0].lower().startswith("sentence"):
                sentence_id = f"{parts[0]} {parts[1]}"
            elif len(parts) >= 3:
                sentence_id = parts[0]
            else:
                sentence_id = None
            token = parts[-2] if len(parts) >= 3 else parts[0]
            tag = parts[-1]
            if tag not in LABEL2ID:
                raise ValueError(f"Unknown tag {tag!r} at line {line_number}")

            if sentence_id is not None and previous_sentence_id is not None and sentence_id != previous_sentence_id:
                flush_sentence()
            if sentence_id is not None:
                previous_sentence_id = sentence_id

            tokens.append(token)
            tags.append(tag)

    flush_sentence()

    if not sentences:
        raise ValueError(f"No sentences found in {file_path}")
    if len(sentences) < 3:
        raise ValueError(
            f"Dataset was parsed into only {len(sentences)} sentence(s). "
            "Check whether the file uses sentence_id/word/tag columns or blank-line sentence separators."
        )

    return sentences


def encode_tags(records: Iterable[dict[str, list[str]]]) -> list[dict[str, list[int] | list[str]]]:
    """Convert string BIO tags to integer ids."""

    encoded = []
    for row in records:
        encoded.append(
            {
                "tokens": row["tokens"],
                "ner_tags": [LABEL2ID[tag] for tag in row["ner_tags"]],
            }
        )
    return encoded


def split_records(
    records: list[dict[str, list[int] | list[str]]],
    test_size: float = 0.1,
    validation_size: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split sentence records into train, validation, and test partitions."""

    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1")

    train_records, temp_records = train_test_split(
        records,
        test_size=test_size + validation_size,
        random_state=seed,
        shuffle=True,
    )
    relative_test_size = test_size / (test_size + validation_size)
    validation_records, test_records = train_test_split(
        temp_records,
        test_size=relative_test_size,
        random_state=seed,
        shuffle=True,
    )
    return train_records, validation_records, test_records


def align_labels_with_tokens(labels: list[int], word_ids: list[int | None]) -> list[int]:
    """Align word-level BIO labels to tokenizer subword positions."""

    aligned_labels: list[int] = []
    previous_word_id: int | None = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            aligned_labels.append(labels[word_id])
        else:
            aligned_labels.append(-100)
        previous_word_id = word_id

    return aligned_labels
