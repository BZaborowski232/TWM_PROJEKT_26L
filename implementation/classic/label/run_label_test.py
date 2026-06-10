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



from detect_damaged_label import (
    detect_damaged_label,
    draw_damaged_label_result
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
    / "label"
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

    if 6 in found_classes:
        return "damaged_label"

    return "good"



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


        img_h, img_w = image.shape[:2]

        # --------------------------------
        # Ground truth
        # --------------------------------

        label_path = (
            LABELS_DIR
            / f"{img_path.stem}.txt"
        )

        bottle_bbox = get_bottle_bbox(
            label_path,
            img_w,
            img_h
        )



        true_label = get_true_label(
            label_path
        )


        label_result = detect_damaged_label(
            image,
            bottle_bbox
        )

        predicted_label = (
            "damaged_label"
            if label_result["damaged"]
            else "good"
        )

        # --------------------------------
        # Wizualizacja
        # --------------------------------

        annotated = draw_damaged_label_result(
        image,
        label_result
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
            label_result["confidence"],

        # "mean_saturation":
        #     label_result["mean_saturation"],

        # "edge_ratio":
        #     label_result["edge_ratio"],

        # "std_saturation":
        #     label_result["std_saturation"],

        # "std_gray":
        #     label_result["std_gray"],

        "blue_ratio":
            label_result["blue_ratio"],

        "largest_blue_ratio":
            label_result["largest_blue_ratio"],

    })

    # --------------------------------
    # CSV
    # --------------------------------

    df = pd.DataFrame(rows)

    results_csv = (
        OUTPUT_DIR
        / "label_results.csv"
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
            "damaged_label"
        ],
        title="DAMAGED LABEL DETECTION RESULTS"
    )

    print("\n===== LABEL RESULTS =====")

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
        "\nLabel test zakończony!"
    )


if __name__ == "__main__":
    main()