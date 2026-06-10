
from pathlib import Path
from ultralytics import YOLO


# ścieżka do głównego katalogu projektu
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ścieżka do wytrenowanego modelu (wynik treningu)
MODEL_PATH = PROJECT_ROOT / "outputs" / "yolo_train" / "weights" / "best.pt"

# katalog ze zdjęciami testowymi
SOURCE_DIR = PROJECT_ROOT / "dataset_yolo" / "test" / "images"

TEST_LABELS_DIR = (
    PROJECT_ROOT
    / "dataset_yolo"
    / "test"
    / "labels"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "yolo_predictions"
)

from yolo.evaluate_yolo_prediction import (
    evaluate_yolo_predictions
)

# używa wytrenowanego modelu do predykcji na obrazach testowych
def main():
    # sprawdź czy model istnieje
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {MODEL_PATH}\n"
            f"Najpierw uruchom train_yolo.py"
        )

    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=str(SOURCE_DIR),  # folder ze zdjęciami
        imgsz=640,               # rozmiar wejściowy obrazu
        conf=0.25,               # próg pewności detekcji
        save=True,               # zapis obrazów z bboxami
        save_txt=True,           # zapis wyników do plików .txt
        save_conf=True,          # zapis pewności predykcji
        project=str(PROJECT_ROOT / "outputs"),  # katalog wyników
        name="yolo_predictions", # folder wynikowy
        exist_ok=True            # nie nadpisuje błędem istniejącego folderu
    )



    print(
    "\nEvaluating predictions..."
)

    metrics = evaluate_yolo_predictions(

        test_labels_dir=
        TEST_LABELS_DIR,

        pred_labels_dir=
        PREDICTIONS_DIR
        / "labels",

        output_dir=
        PREDICTIONS_DIR
    )

    print(
        "\n===== YOLO RESULTS ====="
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


# uruchomienie skryptu
if __name__ == "__main__":
    main()