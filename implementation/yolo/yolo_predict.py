from pathlib import Path
from ultralytics import YOLO


# ścieżka do głównego katalogu projektu
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ścieżka do wytrenowanego modelu (wynik treningu)
MODEL_PATH = PROJECT_ROOT / "outputs" / "yolo_train" / "weights" / "best.pt"

# katalog ze zdjęciami testowymi
SOURCE_DIR = PROJECT_ROOT / "dataset_yolo" / "test" / "images"

# Ta implementacja wykorzystuje wytrenowany model YOLO do detekcji i klasyfikacji czystości wody na zdjęciach testowych. 
# Wyniki są zapisywane w katalogu outputs/yolo_predictions.


# używa wytrenowanego modelu do predykcji na obrazach testowych
def main():
    # sprawdź czy model istnieje
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {MODEL_PATH}\n"
            f"Najpierw uruchom train_yolo.py"
        )

    # wczytanie wytrenowanego modelu
    model = YOLO(str(MODEL_PATH))

    # wykonanie predykcji
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


# uruchomienie skryptu
if __name__ == "__main__":
    main()