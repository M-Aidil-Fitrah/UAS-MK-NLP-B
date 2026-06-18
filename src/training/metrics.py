"""Evaluation helpers for SkillScope ID."""

from __future__ import annotations

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def decode_predictions(predictions, labels, id2label: dict[int, str]):
    """Decode model logits and BIO labels while ignoring masked positions."""

    predicted_ids = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[int(label_id)] for pred_id, label_id in zip(prediction, label) if label_id != -100]
        for prediction, label in zip(predicted_ids, labels)
    ]
    true_predictions = [
        [id2label[int(pred_id)] for pred_id, label_id in zip(prediction, label) if label_id != -100]
        for prediction, label in zip(predicted_ids, labels)
    ]

    return true_predictions, true_labels


def build_compute_metrics(id2label: dict[int, str]):
    """Return a Trainer-compatible metrics function."""

    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        true_predictions, true_labels = decode_predictions(predictions, labels, id2label)
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


def build_classification_report(predictions, labels, id2label: dict[int, str]) -> str:
    """Build an entity-level seqeval report."""

    true_predictions, true_labels = decode_predictions(predictions, labels, id2label)
    return classification_report(true_labels, true_predictions, digits=4)
