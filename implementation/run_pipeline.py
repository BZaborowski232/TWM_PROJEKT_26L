""""
Główny plik uruchamiający cały pipeline projektu (YOLO + metoda klasyczna HSV).

🔷 Co robi:
Plik pozwala uruchomić kolejne etapy przetwarzania danych i modelu:
1. Podział datasetu (dataset_split)
2. Sprawdzenie poprawności etykiet (check_datasets)
3. Trening modelu YOLO (yolo_train)
4. Predykcja YOLO na zbiorze testowym (yolo_predict)
5. Klasyczna analiza obrazu (roi_hsv)

Domyślnie uruchamiane są wszystkie kroki po kolei.

---

🔷 Jak używać:

▶ Uruchom cały pipeline:
    python run_pipeline.py

▶ Uruchom tylko wybrany krok:
    python run_pipeline.py --step split
    python run_pipeline.py --step check
    python run_pipeline.py --step train
    python run_pipeline.py --step predict
    python run_pipeline.py --step classical

---

🔷 Opis kroków:

split       → przygotowanie datasetu w formacie YOLO
check       → sprawdzenie datasetu (klasy, puste etykiety)
train       → trening modelu YOLO
predict     → predykcja na obrazach testowych
classical   → analiza klasyczna (HSV, bez AI)

---
"""


import argparse
import sys
from pathlib import Path

# 🔷 dodaj główny katalog do ścieżki Pythona
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# 🔷 importy z Twojej struktury
from utils.dataset_split import main as split_data
from utils.check_datasets import main as check_data
from yolo.yolo_train import main as train
from yolo.yolo_predict import main as predict
from classic.roi_hsv import main as classical


def run_all():
    print("\n[1/5] Dataset split...")
    split_data()

    print("\n[2/5] Dataset check...")
    check_data()

    print("\n[3/5] Training YOLO...")
    train()

    print("\n[4/5] YOLO prediction...")
    predict()

    print("\n[5/5] Classical HSV...")
    classical()

    print("\n✅ Pipeline zakończony!")


def main():
    parser = argparse.ArgumentParser(description="Pipeline YOLO + HSV")

    parser.add_argument(
        "--step",
        choices=["all", "split", "check", "train", "predict", "classical"],
        default="all"
    )

    args = parser.parse_args()

    if args.step == "split":
        split_data()

    elif args.step == "check":
        check_data()

    elif args.step == "train":
        train()

    elif args.step == "predict":
        predict()

    elif args.step == "classical":
        classical()

    else:
        run_all()


if __name__ == "__main__":
    main()


