
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# ścieżka do głównego katalogu projektu
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# katalog z obrazami testowymi
INPUT_DIR = PROJECT_ROOT / "dataset_yolo" / "test" / "images"

# katalog na wyniki (obrazy + CSV)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classical_results"

# utwórz folder wynikowy jeśli nie istnieje
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# obsługiwane formaty obrazów
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Ta implementacja klasyfikuje czystość wody na podstawie analizy koloru i jasności w centralnym ROI zdjęcia.

# wybiera centralny fragment obrazu (ROI = region zainteresowania)
def get_center_water_roi(image):

    h, w = image.shape[:2]  # wysokość i szerokość

    # wyznaczenie prostokąta w środku obrazu
    x1 = int(w * 0.40)
    x2 = int(w * 0.60)
    y1 = int(h * 0.35)
    y2 = int(h * 0.82)

    return x1, y1, x2, y2


# analizuje kolor i jasność, żeby określić czystość wody
def analyze_water_cleanliness(image):
    # pobierz ROI
    x1, y1, x2, y2 = get_center_water_roi(image)

    roi = image[y1:y2, x1:x2]  # wytnij fragment obrazu

    # konwersja do przestrzeni HSV (łatwiejsza analiza koloru)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # kanały: hue (kolor), saturation, value (jasność)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    # maska ciemnych pikseli
    dark_mask = v_ch < 85

    # maska "brązowych" pikseli (typowe dla brudnej wody / coli)
    brown_mask = (
        (h_ch >= 5) &
        (h_ch <= 35) &
        (s_ch > 35) &
        (v_ch < 230)
    )

    # obliczenie proporcji pikseli
    dark_ratio = np.count_nonzero(dark_mask) / dark_mask.size
    brown_ratio = np.count_nonzero(brown_mask) / brown_mask.size

    # średnia saturacja i jasność
    mean_saturation = float(np.mean(s_ch))
    mean_value = float(np.mean(v_ch))

    # decyzja na podstawie progów
    if brown_ratio > 0.08 or dark_ratio > 0.20 or mean_saturation > 65:
        decision = "water_dirty"
    else:
        decision = "water_clean"

    # zapis dodatkowych cech (do analizy)
    features = {
        "dark_ratio": dark_ratio,
        "brown_ratio": brown_ratio,
        "mean_saturation": mean_saturation,
        "mean_value": mean_value,
        "roi_x1": x1,
        "roi_y1": y1,
        "roi_x2": x2,
        "roi_y2": y2,
    }

    return decision, features


# rysuje wynik na obrazie (prostokąt + etykieta)
def draw_result(image, decision, features):
    annotated = image.copy()

    # współrzędne ROI
    x1 = features["roi_x1"]
    y1 = features["roi_y1"]
    x2 = features["roi_x2"]
    y2 = features["roi_y2"]

    # kolor: zielony = czysta, czerwony = brudna
    if decision == "water_clean":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    # rysuj prostokąt ROI
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    # dodaj tekst z decyzją
    cv2.putText(
        annotated,
        decision,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    return annotated


def main():
    rows = []  # lista wyników do CSV

    # lista obrazów w folderze
    image_paths = sorted([
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])

    # jeśli brak danych → błąd
    if not image_paths:
        raise RuntimeError(
            f"Nie znaleziono zdjęć w {INPUT_DIR}. "
            f"Najpierw uruchom dataset_split.py."
        )

    # przetwarzanie każdego obrazu
    for img_path in image_paths:
        image = cv2.imread(str(img_path))

        # jeśli nie udało się wczytać → pomiń
        if image is None:
            print(f"Nie udało się wczytać: {img_path.name}")
            continue

        # analiza + wizualizacja
        decision, features = analyze_water_cleanliness(image)
        annotated = draw_result(image, decision, features)

        # zapis obrazu wynikowego
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), annotated)

        # zapis danych do CSV
        row = {
            "image": img_path.name,
            "decision": decision,
            "dark_ratio": features["dark_ratio"],
            "brown_ratio": features["brown_ratio"],
            "mean_saturation": features["mean_saturation"],
            "mean_value": features["mean_value"],
            "roi_x1": features["roi_x1"],
            "roi_y1": features["roi_y1"],
            "roi_x2": features["roi_x2"],
            "roi_y2": features["roi_y2"],
        }

        rows.append(row)

    # zapis wyników do pliku CSV
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "classical_roi_hsv_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # komunikaty końcowe
    print("Gotowe.")
    print(f"Zapisano obrazy z ROI do: {OUTPUT_DIR}")
    print(f"Zapisano CSV do: {csv_path}")


# uruchomienie skryptu
if __name__ == "__main__":
    main()