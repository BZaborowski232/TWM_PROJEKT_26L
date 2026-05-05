from pathlib import Path
from ultralytics import YOLO


# ścieżka do głównego katalogu projektu (2 poziomy wyżej)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# plik konfiguracyjny YOLO (ścieżki do danych + klasy)
DATA_YAML = PROJECT_ROOT / "implementation" / "yolo" / "data.yaml"


# Ta implementacja trenuje model YOLO na przygotowanym zbiorze danych. 
# Model jest zapisywany w katalogu outputs/yolo_train/weights/best.pt, skąd może być później użyty do predykcji.


# trenuje model YOLO i zapisuje go do outputs/yolo_train/
def main():
    # wczytanie pretrenowanego modelu (lekka wersja YOLOv8)
    model = YOLO("yolov8n.pt")

    # rozpoczęcie treningu
    model.train(
        data=str(DATA_YAML),  # dataset + klasy
        epochs=10,            # liczba epok (ile razy model widzi dane)
        imgsz=640,            # rozmiar obrazów
        batch=8,              # ile obrazów na batch
        project=str(PROJECT_ROOT / "outputs"),  # katalog wyników
        name="yolo_train",    # nazwa folderu z treningiem
        exist_ok=True         # nie wywala błędu jeśli folder istnieje
    )


# uruchomienie skryptu
if __name__ == "__main__":
    main()