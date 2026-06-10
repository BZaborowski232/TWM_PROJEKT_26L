from pathlib import Path
import pandas as pd

from classic.common.evaluate import (
    evaluate_classic
)


CLASS_MAP = {
    0: "good",
    1: "wrong_bottle",
    2: "underfilled",
    3: "no_cap",
    4: "loose_cap",
    5: "debris",
    6: "damaged_label"
}


SUPPORTED_CLASSES = [
    "good",
    "wrong_bottle",
    "underfilled",
    "no_cap",
    "loose_cap",
    "debris",
    "damaged_label"
]


def get_image_label(label_path):

    if not label_path.exists():
        return "background"

    found_classes = set()

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if not parts:
                continue

            found_classes.add(
                int(parts[0])
            )

    # priorytet wad

    if 1 in found_classes:
        return "wrong_bottle"

    if 2 in found_classes:
        return "underfilled"

    if 3 in found_classes:
        return "no_cap"

    if 4 in found_classes:
        return "loose_cap"

    if 5 in found_classes:
        return "debris"

    if 6 in found_classes:
        return "damaged_label"

    return "good"


def evaluate_yolo_predictions(
    test_labels_dir,
    pred_labels_dir,
    output_dir
):

    rows = []

    test_labels_dir = Path(
        test_labels_dir
    )

    pred_labels_dir = Path(
        pred_labels_dir
    )

    output_dir = Path(
        output_dir
    )

    for gt_label in sorted(
        test_labels_dir.glob("*.txt")
    ):

        pred_label = (
            pred_labels_dir
            / gt_label.name
        )

        rows.append({

            "image":
                gt_label.stem,

            "true_label":
                get_image_label(
                    gt_label
                ),

            "predicted_label":
                get_image_label(
                    pred_label
                )
        })

    df = pd.DataFrame(rows)

    results_csv = (
        output_dir
        / "yolo_results.csv"
    )

    df.to_csv(
        results_csv,
        index=False,
        encoding="utf-8-sig"
    )

    metrics = evaluate_classic(
        results_csv=results_csv,
        output_dir=output_dir,
        supported_classes=SUPPORTED_CLASSES,
        title="YOLO PREDICTION RESULTS"
    )

    return metrics