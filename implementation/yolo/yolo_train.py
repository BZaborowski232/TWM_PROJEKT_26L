from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "implementation" / "yolo" / "data.yaml"


# Ta implementacja trenuje model YOLO na przygotowanym zbiorze danych. Model jest zapisywany w katalogu outputs/yolo_train/weights/best.pt, skąd może być później użyty do predykcji.


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=640,
        batch=8,
        project=str(PROJECT_ROOT / "outputs"),
        name="yolo_train",
        exist_ok=True
    )


if __name__ == "__main__":
    main()