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



from detect_debris import (
    detect_debris,
    draw_debris_result
)

from common.evaluate import (
    evaluate_classic
)




# --------------------------------------------------
# Ścieżki
# --------------------------------------------------



PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    / "debris"
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

    if 5 in found_classes:
        return "debris"

    return "good"


# --------------------------------------------------
# Test debris
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
        # Ground truth
        # --------------------------------

        label_path = (
            LABELS_DIR
            / f"{img_path.stem}.txt"
        )

        true_label = get_true_label(
            label_path
        )

        # --------------------------------
        # Detekcja debris
        # --------------------------------

        debris_result = detect_debris(
            image
        )

        predicted_label = (
            "debris"
            if debris_result["debris"]
            else "good"
        )

        # --------------------------------
        # Wizualizacja
        # --------------------------------

        annotated = draw_debris_result(
            image,
            debris_result
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

            "confidence":
                debris_result["confidence"],

            "brown_ratio":
                debris_result["brown_ratio"],

            "dark_ratio":
                debris_result["dark_ratio"],

            "mean_saturation":
                debris_result["mean_saturation"],

            "mean_value":
                debris_result["mean_value"]
        })

    # --------------------------------
    # CSV
    # --------------------------------

    df = pd.DataFrame(rows)

    results_csv = (
        OUTPUT_DIR
        / "debris_results.csv"
    )

    df.to_csv(
        results_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nCSV zapisano:\n{results_csv}"
    )

    # --------------------------------
    # Ewaluacja
    # --------------------------------

    metrics = evaluate_classic(
        results_csv=results_csv,
        output_dir=OUTPUT_DIR,
        supported_classes=[
            "good",
            "debris"
        ],
        title="DEBRIS DETECTION RESULTS"
    )

    print("\n===== DEBRIS RESULTS =====")

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
        "\nDebris test zakończony!"
    )


if __name__ == "__main__":
    main()