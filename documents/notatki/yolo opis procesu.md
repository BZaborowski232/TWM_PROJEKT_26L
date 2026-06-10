# Proces działania modułu YOLO

## Cel modułu

Moduł YOLO odpowiada za automatyczne wykrywanie i klasyfikację wad butelek na podstawie obrazów. W przeciwieństwie do algorytmów klasycznych, model YOLO samodzielnie lokalizuje obiekt na obrazie oraz przypisuje mu odpowiednią klasę.

---

# Struktura modułu

```text
implementation/
│
├── yolo/
│   ├── data.yaml
│   ├── yolo_train.py
│   └── yolo_predict.py
│
├── utils/
│   ├── dataset_split.py
│   └── check_datasets.py
│
└── run_pipeline_yolo.py
```

---

# Etap 1 – Przygotowanie zbioru danych

Proces rozpoczyna się od przygotowania danych wejściowych.

Oryginalny zbiór posiada strukturę:

```text
images/
├── train/
└── val/

labels/
├── train/
└── val/
```

Każdy obraz posiada odpowiadający mu plik etykiety YOLO:

```text
image_001.jpg
image_001.txt
```

Przykładowa etykieta YOLO:

```text
0 0.52 0.47 0.18 0.64
```

gdzie:

* 0 – identyfikator klasy
* 0.52 – współrzędna X środka obiektu
* 0.47 – współrzędna Y środka obiektu
* 0.18 – szerokość obiektu
* 0.64 – wysokość obiektu

Współrzędne zapisane są w postaci znormalizowanej (0–1).

---

# Etap 2 – Podział danych

Skrypt:

```bash
python dataset_split.py
```

dzieli dane na:

```text
dataset_yolo/
├── train/
├── val/
└── test/
```

Podział wykonywany jest następująco:

* 80% obrazów treningowych trafia do train
* 20% obrazów treningowych trafia do val
* oryginalny zbiór walidacyjny zostaje wykorzystany jako test

Dzięki temu możliwe jest niezależne trenowanie, walidacja oraz końcowa ocena modelu.

---

# Etap 3 – Weryfikacja zbioru

Przed rozpoczęciem treningu wykonywany jest skrypt:

```bash
python check_datasets.py
```

Jego zadaniem jest:

* sprawdzenie liczby plików etykiet
* wykrycie pustych plików etykiet
* zliczenie wystąpień poszczególnych klas

Pozwala to wykryć błędy w zbiorze danych przed treningiem modelu.

---

# Etap 4 – Konfiguracja modelu

Plik:

```text
data.yaml
```

zawiera informacje o:

* lokalizacji zbiorów train, val oraz test
* liczbie klas
* nazwach klas

Przykładowo:

```yaml
names:
  0: good
  3: no_cap
  5: debris
  6: damaged_label
```

Na podstawie tego pliku YOLO wie jakie klasy ma rozpoznawać.

---

# Etap 5 – Trening modelu

Trening uruchamiany jest poleceniem:

```bash
python yolo_train.py
```

Wykorzystywany jest model bazowy:

```text
yolov8n.pt
```

czyli lekka wersja YOLOv8 Nano.

Podczas treningu model analizuje obrazy ze zbioru train i stopniowo aktualizuje swoje wagi.

Dla każdej epoki wykonywane są dwa kroki:

```text
TRAIN
↓
aktualizacja wag

VALIDATION
↓
ocena modelu
```

Zbiór walidacyjny nie bierze udziału w uczeniu i służy jedynie do monitorowania jakości modelu.

---

# Rola zbioru Validation

Validation pozwala sprawdzić czy model rzeczywiście uczy się rozpoznawania obiektów, a nie jedynie zapamiętuje dane treningowe.

Przykład:

```text
Train Accuracy = 99%
Validation Accuracy = 60%
```

oznacza overfitting, czyli przeuczenie modelu.

Najlepszy model wybierany jest na podstawie wyników uzyskanych właśnie na zbiorze validation.

---

# Wyniki treningu

Po zakończeniu treningu generowany jest katalog:

```text
outputs/
└── yolo_train/
    └── weights/
        ├── best.pt
        └── last.pt
```

gdzie:

* best.pt – model o najlepszych wynikach walidacyjnych
* last.pt – model z ostatniej epoki

W dalszej części projektu wykorzystywany jest model:

```text
best.pt
```

---

# Etap 6 – Predykcja

Po wytrenowaniu modelu wykonywany jest:

```bash
python yolo_predict.py
```

Model:

```text
best.pt
```

analizuje obrazy znajdujące się w:

```text
dataset_yolo/test/images
```

Dla każdego obrazu YOLO wykonuje:

```text
Obraz
↓
Detekcja obiektu
↓
Wyznaczenie bbox
↓
Klasyfikacja
↓
Wynik
```

Przykład:

```text
no_cap
confidence = 0.94
```

---

# Zapisywanie wyników

Wyniki predykcji trafiają do:

```text
outputs/yolo_predictions/
```

Zapisywane są:

* obrazy z narysowanymi bounding boxami
* pliki tekstowe z predykcjami
* wartości confidence

Pozwala to na późniejszą analizę skuteczności modelu.

---

# Pełny przepływ działania

```text
Oryginalny zbiór danych
        │
        ▼
dataset_split.py
        │
        ▼
dataset_yolo/
(train / val / test)
        │
        ▼
check_datasets.py
        │
        ▼
data.yaml
        │
        ▼
yolo_train.py
        │
        ▼
best.pt
        │
        ▼
yolo_predict.py
        │
        ▼
outputs/yolo_predictions/
```

---

# Porównanie z algorytmami klasycznymi

YOLO:

```text
Obraz
↓
Lokalizacja obiektu
↓
Klasyfikacja
↓
Wynik
```

Algorytmy klasyczne:

```text
Obraz
+
bbox z etykiety YOLO
↓
Analiza ROI
↓
Klasyfikacja
↓
Wynik
```

Oznacza to, że YOLO rozwiązuje jednocześnie problem lokalizacji oraz klasyfikacji, natomiast algorytmy klasyczne wykorzystują wcześniej znaną lokalizację obiektu i skupiają się wyłącznie na wykrywaniu defektu.
