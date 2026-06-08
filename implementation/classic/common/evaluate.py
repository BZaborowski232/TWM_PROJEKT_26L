from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


def evaluate(
    results_csv,
    supported_classes
):

    df = pd.read_csv(results_csv)

    df = df[
        df["true_label"].isin(
            supported_classes
        )
    ]

    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report
    }


def save_confusion_matrix(
    y_true,
    y_pred,
    output_path,
    supported_classes
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=supported_classes
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=supported_classes
    )

    disp.plot()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def evaluate_classic(
    results_csv,
    output_dir,
    supported_classes,
    title
):

    df = pd.read_csv(results_csv)

    df = df[
        df["true_label"].isin(
            supported_classes
        )
    ]

    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    metrics = evaluate(
        results_csv,
        supported_classes
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        Path(output_dir)
        / "confusion_matrix.png",
        supported_classes
    )

    errors = df[
        df["true_label"]
        !=
        df["predicted_label"]
    ]

    errors.to_csv(
        Path(output_dir)
        / "misclassified.csv",
        index=False,
        encoding="utf-8-sig"
    )

    metrics_path = (
        Path(output_dir)
        / "metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"{title}\n\n"
        )

        f.write(
            f"Accuracy : {metrics['accuracy']:.4f}\n"
        )

        f.write(
            f"Precision: {metrics['precision']:.4f}\n"
        )

        f.write(
            f"Recall   : {metrics['recall']:.4f}\n"
        )

        f.write(
            f"F1-score : {metrics['f1']:.4f}\n\n"
        )

        f.write(
            "Classification Report\n\n"
        )

        f.write(
            metrics["report"]
        )

    return metrics