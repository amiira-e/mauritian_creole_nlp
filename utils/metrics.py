"""
Evaluation metrics for Mauritian Creole joint text restoration task.

Used by:
- BiLSTM
- T5-small
- ByT5-small
- Qwen2.5-0.5B
"""

import numpy as np

from jiwer import wer, cer
from sacrebleu.metrics import CHRF

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
)

# chrF metric
chrf_metric = CHRF()


def exact_match(predictions, references):
    """
    Percentage of sentences exactly matching.
    """

    correct = sum(
        p == r
        for p, r in zip(predictions, references)
    )

    return correct / len(references)


def word_accuracy(predictions, references):
    """
    Word-level accuracy for normalization.
    """

    correct_words = 0
    total_words = 0

    for pred, ref in zip(predictions, references):

        pred_words = pred.split()
        ref_words = ref.split()

        for p, r in zip(pred_words, ref_words):

            if p == r:
                correct_words += 1

        total_words += len(ref_words)

    return correct_words / total_words


def punctuation_metrics(predictions, references):
    """
    Precision, recall and F1 for punctuation restoration.
    """

    punctuation = ".,!?;:"

    pred_labels = []
    ref_labels = []


    for pred, ref in zip(predictions, references):

        pred_labels.extend(
            [1 if c in punctuation else 0 for c in pred]
        )

        ref_labels.extend(
            [1 if c in punctuation else 0 for c in ref]
        )


    precision, recall, f1, _ = precision_recall_fscore_support(
        ref_labels,
        pred_labels,
        average="binary",
        zero_division=0
    )

    return {
        "punct_precision": precision,
        "punct_recall": recall,
        "punct_f1": f1
    }


def capitalization_metrics(predictions, references):
    """
    Accuracy and F1 for capitalization restoration.
    """

    pred_labels = []
    ref_labels = []


    for pred, ref in zip(predictions, references):

        for p, r in zip(pred, ref):

            if p.isalpha() and r.isalpha():

                pred_labels.append(
                    int(p.isupper())
                )

                ref_labels.append(
                    int(r.isupper())
                )


    precision, recall, f1, _ = precision_recall_fscore_support(
        ref_labels,
        pred_labels,
        average="binary",
        zero_division=0
    )

    return {
        "capitalization_accuracy":
            accuracy_score(
                ref_labels,
                pred_labels
            ),

        "capitalization_f1":
            f1
    }


def calculate_all_metrics(predictions, references):
    """
    Calculate all evaluation metrics.
    """


    results = {}


    # Overall metrics

    results["CER"] = cer(
        references,
        predictions
    )


    results["WER"] = wer(
        references,
        predictions
    )


    results["chrF"] = chrf_metric.corpus_score(
        predictions,
        [references]
    ).score


    results["Exact_Match"] = exact_match(
        predictions,
        references
    )


    # Normalization metrics

    results["Normalization_CER"] = results["CER"]

    results["Normalization_WER"] = results["WER"]

    results["Word_Accuracy"] = word_accuracy(
        predictions,
        references
    )


    # Punctuation

    results.update(
        punctuation_metrics(
            predictions,
            references
        )
    )


    # Capitalization

    results.update(
        capitalization_metrics(
            predictions,
            references
        )
    )

    return results