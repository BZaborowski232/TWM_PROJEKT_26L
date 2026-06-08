# System Inspekcji Butelek – Instrukcja

## Opis modułów

### `common/bottle_detector.py`

Wspólny moduł wykorzystywany przez algorytmy klasyczne.

Zawiera funkcję:

```python
get_bottle_bbox(...)
```

Funkcja odczytuje bounding box butelki z pliku etykiet YOLO i zwraca:

```python
(bx, by, bw, bh)
```

Dzięki temu wszystkie algorytmy mogą pracować na obszarze rzeczywiście zawierającym butelkę.

---

### `common/evaluate.py`

Uniwersalny moduł ewaluacji dla wszystkich klasycznych detektorów.

Generuje:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* `misclassified.csv`

---

### `cap/detect_cap.py`

Klasyczna detekcja obecności zakrętki.

Wykorzystuje:

* bbox butelki
* analizę koloru zakrętki w przestrzeni HSV
* analizę rozmiaru wykrytego obszaru

Klasy:

* `good`
* `no_cap`

---

### `debris/detect_debris.py`

Klasyczna detekcja zanieczyszczeń.

Wykorzystuje:

* analizę kolorów
* analizę ciemnych obszarów
* analizę ROI

Klasy:

* `good`
* `debris`

---

### `label/detect_damaged_label.py`

Klasyczna detekcja uszkodzonej lub brakującej etykiety.

Wykorzystuje:

* ROI etykiety wyznaczone na podstawie bbox butelki
* średnie nasycenie (`mean_saturation`)
* odchylenie standardowe nasycenia (`std_saturation`)
* odchylenie standardowe jasności (`std_gray`)

Klasy:

* `good`
* `damaged_label`

---

### `fill_level/detect_fill_level.py`

Moduł przeznaczony do wykrywania poziomu napełnienia butelki.

Obecnie znajduje się w trakcie rozwoju.

---

### `yolo/yolo_train.py`

Skrypt do trenowania modelu YOLO.

---

### `yolo/yolo_predict.py`

Skrypt do wykonywania predykcji przy użyciu wytrenowanego modelu YOLO.

---

## Uruchamianie testów dla klasycznych metod

Przed uruchomieniem przejść do katalogu:

```bash
implementation/classic
```

### Detekcja zakrętki

```bash
python cap/run_cap_test.py
```

### Detekcja zanieczyszczeń

```bash
python debris/run_debris_test.py
```

### Detekcja etykiety

```bash
python label/run_label_test.py
```

---

## Dane testowe

Obrazy:

```text
dataset_yolo/test/images
```

Etykiety:

```text
dataset_yolo/test/labels
```

---

## Wyniki

Wszystkie wyniki klasycznych metod zapisywane są do:

```text
outputs/classical_results/
```

Przykładowe katalogi:

```text
outputs/classical_results/cap/
outputs/classical_results/debris/
outputs/classical_results/label/
```

Każdy moduł generuje:

* `annotated/` – obrazy z wizualizacją detekcji
* `*_results.csv` – szczegółowe wyniki
* `confusion_matrix.png`
* `metrics.txt`
* `misclassified.csv`

Wszystkie wyniki z yolo zapisywane są do:

```text
outputs/yolo_predictions/
```

---

## Klasy YOLO

| ID | Klasa         |
| -- | ------------- |
| 0  | good          |
| 3  | no_cap        |
| 5  | debris        |
| 6  | damaged_label |

Pozostałe klasy zgodnie z aktualną konfiguracją datasetu.

---

## YOLO

### Trenowanie

```bash
python yolo/yolo_train.py
```

### Predykcja

```bash
python yolo/yolo_predict.py
```

Domyślnie wykorzystywany model:

```text
yolo26n.pt
```

---

## Uwagi

* Obecnie klasyczne algorytmy cap i label wykorzystują bbox butelki zapisany w etykietach YOLO.
* Jeżeli bbox nie zostanie znaleziony, algorytmy przechodzą na domyślnie zdefiniowany ROI.
* Przed uruchomieniem należy upewnić się, że dla każdego obrazu istnieje odpowiadający mu plik `.txt`.
* Wyniki ewaluacji są generowane automatycznie po zakończeniu testu.
