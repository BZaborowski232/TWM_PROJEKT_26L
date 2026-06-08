"""
Test modułu detect_cap

Uruchomienie:

python run_cap_test.py
"""

import cv2
import pandas as pd
import sys
from pathlib import Path


CLASSIC_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(CLASSIC_ROOT)
)


from detect_cap import (
    detect_cap,
    draw_cap_result
)

from common.evaluate import (
    evaluate_classic
)

from common.bottle_detector import (
    get_bottle_bbox
)

# --------------------------------------------------
# Ścieżki
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

print(
    "PROJECT_ROOT:",
    PROJECT_ROOT
)

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

print(
    "INPUT_DIR:",
    INPUT_DIR
)

print(
    "LABELS_DIR:",
    LABELS_DIR
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classical_results"
    / "cap"
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





# --------------------------------------------------
# YOLO classes
# --------------------------------------------------

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

    if 3 in found_classes:
        return "no_cap"

    return "good"





# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    rows = []

    image_paths = sorted([

        p for p in INPUT_DIR.iterdir()

        if p.suffix.lower()
        in IMG_EXTENSIONS
    ])

    if not image_paths:

        raise RuntimeError(
            f"Brak obrazów: {INPUT_DIR}"
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

        img_h, img_w = image.shape[:2]

        label_path = (
            LABELS_DIR
            / f"{img_path.stem}.txt"
        )



        # -------------------------
        # Ground truth
        # -------------------------

        label_path = (
            LABELS_DIR
            / f"{img_path.stem}.txt"
        )

        bottle_bbox = get_bottle_bbox(
            label_path,
            img_w,
            img_h
        )

        if bottle_bbox is None:
            print("NO BBOX:", img_path.name)

        true_label = (
            get_true_label(
                label_path
            )
        )


        # -------------------------
        # Detection
        # -------------------------

        cap_result = detect_cap(
            image,
            bottle_bbox
        )

        predicted_label = (
            "good"
            if cap_result[
                "cap_present"
            ]
            else "no_cap"
        )

        # -------------------------
        # Visualization
        # -------------------------

        annotated = (
            draw_cap_result(
                image,
                cap_result
            )
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

        # -------------------------
        # CSV
        # -------------------------

        rows.append({

            "image":
                img_path.name,

            "true_label":
                true_label,

            "predicted_label":
                predicted_label,

            "cap_present":
                cap_result["cap_present"],

            "confidence":
                cap_result["confidence"],

            "cap_height_ratio":
                cap_result["cap_height_ratio"],

            "cap_width_ratio":
                cap_result["cap_width_ratio"],

            "cap_area_ratio":
                cap_result["cap_area_ratio"],

            "largest_area":
                cap_result["largest_area"]
        })

    # --------------------------------------------------
    # CSV
    # --------------------------------------------------

    df = pd.DataFrame(
        rows
    )

    results_csv = (
        OUTPUT_DIR
        / "cap_results.csv"
    )

    df.to_csv(
        results_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nCSV zapisano:"
        f"\n{results_csv}"
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    metrics = evaluate_classic(
        results_csv=results_csv,
        output_dir=OUTPUT_DIR,
        supported_classes=[
            "good",
            "no_cap"
        ],
        title="CAP DETECTION RESULTS"
    )

    print(
        "\n===== RESULTS ====="
    )

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
        "\n✅ CAP TEST FINISHED"
    )


if __name__ == "__main__":
    main()