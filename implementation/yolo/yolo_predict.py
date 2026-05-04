from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "outputs" / "yolo_train" / "weights" / "best.pt"
SOURCE_DIR = PROJECT_ROOT / "dataset_yolo" / "test" / "images"

# Ta implementacja wykorzystuje wytrenowany model YOLO do detekcji i klasyfikacji czystości wody na zdjęciach testowych. 
# Wyniki są zapisywane w katalogu outputs/yolo_predictions.


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {MODEL_PATH}\n"
            f"Najpierw uruchom train_yolo.py"
        )

    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=str(SOURCE_DIR),
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(PROJECT_ROOT / "outputs"),
        name="yolo_predictions",
        exist_ok=True
    )


if __name__ == "__main__":
    main()