"""
Pipeline klasycznej analizy obrazu

Kroki:
1. Detect debris
2. Detect fill level
3. Detect cap
4. Final classification
5. Save results
6. Evaluate

Uruchomienie:
python run_pipeline_classic.py
"""

import cv2
import pandas as pd
from pathlib import Path

from implementation.classic.debris.detect_debris import (
    detect_debris,
    draw_debris_result
)

from implementation.classic.fill_level.detect_fill_level import (
    detect_fill_level,
    draw_fill_result
)

from implementation.classic.cap.detect_cap import (
    detect_cap,
    draw_cap_result
)

from classic.classify_bottle import (
    classify_bottle
)

from classic.evaluate_classic import (
    evaluate_classic
)


# --------------------------------------------------
# Ścieżki
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "dataset_yolo"
    / "test"
    / "images"
)

LABELS_DIR = (
    PROJECT_ROOT
    / "dataset_yolo"
    / "test"
    / "labels"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classical_results"
)

ANNOTATED_DIR = (
    OUTPUT_DIR
    / "annotated"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANNOTATED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

print("PROJECT_ROOT:", PROJECT_ROOT)
print("INPUT_DIR:", INPUT_DIR)
print("LABELS_DIR:", LABELS_DIR)


# --------------------------------------------------
# Klasy YOLO
# --------------------------------------------------

CLASS_MAP = {
    0: "good",
    # 1: "wrong_bottle",
    # 2: "underfilled",
    3: "no_cap",
    # 4: "loose_cap",
    # 5: "debris",
    # 6: "damaged_label"
}


def get_true_label(label_path):

    if not label_path.exists():
        return "unknown"

    found_classes = set()

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if not parts:
                continue

            found_classes.add(
                int(parts[0])
            )

    PRIORITY = [
        # 5,  # debris
        # 2,  # underfilled
        3,  # no_cap
        # 4,  # loose_cap
        # 6,  # damaged_label
        # 1,  # wrong_bottle
        0   # good
    ]

    for cls_id in PRIORITY:

        if cls_id in found_classes:
            return CLASS_MAP[cls_id]

    return "good"


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

def main():

    rows = []

    image_paths = sorted([
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])

    if not image_paths:

        raise RuntimeError(
            f"Brak obrazów w {INPUT_DIR}"
        )

    print(
        f"\nPrzetwarzanie "
        f"{len(image_paths)} obrazów..."
    )

    for img_path in image_paths:

        image = cv2.imread(
            str(img_path)
        )

        if image is None:
            continue

        # --------------------------------
        # Ground Truth
        # --------------------------------

        label_path = (
            LABELS_DIR
            / f"{img_path.stem}.txt"
        )

        true_label = get_true_label(
            label_path
        )

        # --------------------------------
        # Detektory
        # --------------------------------

        debris_result = detect_debris(
            image
        )

        fill_result = detect_fill_level(
            image
        )

        cap_result = detect_cap(
            image
        )

        # --------------------------------
        # Klasyfikacja końcowa
        # --------------------------------

        classification = classify_bottle(
            debris_result,
            fill_result,
            cap_result
        )

        predicted_label = (
            classification["final_class"]
        )

        # --------------------------------
        # Wizualizacja
        # --------------------------------

        annotated = draw_debris_result(
            image,
            debris_result
        )

        annotated = draw_fill_result(
            annotated,
            fill_result
        )

        annotated = draw_cap_result(
            annotated,
            cap_result
        )

        cv2.putText(
            annotated,
            f"FINAL: {predicted_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3
        )

        cv2.imwrite(
            str(
                ANNOTATED_DIR
                / img_path.name
            ),
            annotated
        )

        # --------------------------------
        # CSV
        # --------------------------------

        rows.append({

            "image":
                img_path.name,

            "true_label":
                true_label,

            "predicted_label":
                predicted_label,

            "prediction_confidence":
                classification["confidence"],

            "decision_source":
                classification["decision_source"],

            # debris

            "debris":
                debris_result["debris"],

            "debris_conf":
                debris_result["confidence"],

            "brown_ratio":
                debris_result["brown_ratio"],

            "dark_ratio":
                debris_result["dark_ratio"],

            "mean_saturation":
                debris_result["mean_saturation"],

            # fill

            "underfilled":
                fill_result["underfilled"],

            "fill_conf":
                fill_result["confidence"],

            "fill_ratio":
                fill_result["fill_ratio"],

            # cap

            "cap_present":
                cap_result["cap_present"],

            "cap_conf":
                cap_result["confidence"],

            "blue_ratio":
                cap_result["blue_ratio"],

            "cap_height_ratio":
            cap_result["cap_height_ratio"]
        })

    # --------------------------------------------------
    # CSV
    # --------------------------------------------------

    df = pd.DataFrame(rows)

    results_csv = (
        OUTPUT_DIR
        / "classical_results.csv"
    )

    df.to_csv(
        results_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nCSV zapisano:\n{results_csv}"
    )

    # --------------------------------------------------
    # Ewaluacja
    # --------------------------------------------------

    metrics = evaluate_classic(
        results_csv,
        OUTPUT_DIR
    )

    print("\n===== RESULTS =====")

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score : "
        f"{metrics['f1']:.4f}"
    )

    print(
        "\nClassical pipeline zakończony!"
    )


if __name__ == "__main__":
    main()