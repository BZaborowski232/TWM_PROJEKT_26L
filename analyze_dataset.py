from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Ścieżki
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset_yolo"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_analysis"
)

OUTPUT_DIR.mkdir(
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
# Klasy
# --------------------------------------------------

CLASSES = [

    "good",
    "wrong_bottle",
    "underfilled",
    "no_cap",
    "loose_cap",
    "debris",
    "damaged_label"
]


# --------------------------------------------------
# Odczyt klasy z nazwy pliku
# --------------------------------------------------

def get_class_from_filename(filename):

    name = filename.lower()

    for cls in CLASSES:

        if name.startswith(cls):
            return cls

    return None


# --------------------------------------------------
# Analiza jednego zbioru
# --------------------------------------------------

def analyze_split(split_name):

    images_dir = (
        DATASET_DIR
        / split_name
        / "images"
    )

    counter = Counter()

    total_images = 0

    for img_path in images_dir.iterdir():

        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue

        total_images += 1

        cls = get_class_from_filename(
            img_path.stem
        )

        if cls is not None:
            counter[cls] += 1

    rows = []

    for cls in CLASSES:

        count = counter.get(
            cls,
            0
        )

        percentage = (
            count
            / total_images
            * 100
        ) if total_images > 0 else 0

        rows.append({

            "class_name":
                cls,

            "count":
                count,

            "percentage":
                round(
                    percentage,
                    2
                ),

            "split":
                split_name
        })

    df = pd.DataFrame(rows)

    csv_path = (
        OUTPUT_DIR
        / f"{split_name}_distribution.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------------
    # Wykres
    # ----------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    bars = plt.bar(
        df["class_name"],
        df["count"]
    )

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom"
        )

    plt.title(
        f"Class Distribution ({split_name})"
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Number of images"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{split_name}_distribution.png",
        dpi=300
    )

    plt.close()

    return df, total_images


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    all_data = []

    print(
        "\n===== DATASET ANALYSIS ====="
    )

    for split in [

        "train",
        "val",
        "test"

    ]:

        df, total_images = (
            analyze_split(split)
        )

        all_data.append(df)

        print(
            f"\n{split.upper()}: "
            f"{total_images} images"
        )

        print(
            df[
                [
                    "class_name",
                    "count",
                    "percentage"
                ]
            ]
        )

    full_df = pd.concat(
        all_data,
        ignore_index=True
    )

    full_df.to_csv(

        OUTPUT_DIR
        / "all_distributions.csv",

        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()