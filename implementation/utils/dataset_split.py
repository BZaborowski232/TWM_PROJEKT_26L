from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

#ten plik wypełnmia katalog dataset_yolo obrazami i etykietami w strukturze zgodnej z wymaganiami YOLO. Dzieli oryginalny zbiór treningowy na nowy train i val, a oryginalny zbiór walidacyjny przenosi do test. Dzięki temu można łatwo trenować model YOLO na nowym zbiorze, a jednocześnie zachować oryginalny zbiór walidacyjny jako testowy.


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_IMAGES_TRAIN = PROJECT_ROOT / "images" / "train"
SRC_LABELS_TRAIN = PROJECT_ROOT / "labels" / "train"

SRC_IMAGES_VAL = PROJECT_ROOT / "images" / "val"
SRC_LABELS_VAL = PROJECT_ROOT / "labels" / "val"

OUT = PROJECT_ROOT / "dataset_yolo"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def recreate_output_dirs():
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            path = OUT / split / subdir

            if path.exists():
                shutil.rmtree(path)

            path.mkdir(parents=True, exist_ok=True)


def copy_image_and_label(img_path: Path, src_label_dir: Path, split: str):
    label_path = src_label_dir / f"{img_path.stem}.txt"

    dst_img = OUT / split / "images" / img_path.name
    dst_label = OUT / split / "labels" / label_path.name

    shutil.copy2(img_path, dst_img)

    if label_path.exists():
        shutil.copy2(label_path, dst_label)
    else:
        print(f"UWAGA: brak etykiety dla obrazu: {img_path.name}")


def get_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Nie znaleziono folderu: {folder}")

    return sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])


def main():
    recreate_output_dirs()

    train_images = get_images(SRC_IMAGES_TRAIN)

    new_train, new_val = train_test_split(
        train_images,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    for img_path in new_train:
        copy_image_and_label(img_path, SRC_LABELS_TRAIN, "train")

    for img_path in new_val:
        copy_image_and_label(img_path, SRC_LABELS_TRAIN, "val")

    original_val_images = get_images(SRC_IMAGES_VAL)

    for img_path in original_val_images:
        copy_image_and_label(img_path, SRC_LABELS_VAL, "test")

    print("Gotowe. Utworzono dataset_yolo.")
    print(f"Nowy train: {len(new_train)} obrazów")
    print(f"Nowy val:   {len(new_val)} obrazów")
    print(f"Nowy test:  {len(original_val_images)} obrazów")


if __name__ == "__main__":
    main()