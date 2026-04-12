# TWM - Techniki Widzenia Maszynowego - Projekt 26L
## Temat: "Optyczna inspekcja obiektów na taśmociągu"
### Zespół:
* Piotr Walczak
* Katarzyna Wawer
* Bartosz Zaborowski

### ETAP 1 - Założenia wstepne projektu

#### 1. Opis problemu

**Cel projektu:** 
Celem projektu jest stworzenie autorskiego systemu wizyjnego opartego na metodach klasycznego przetwarzania obrazu, którego zadaniem jest automatyczna inspekcja jakości butelek przemieszczających się na symulowanym taśmociągu. System ma za zadanie w czasie rzeczywistym weryfikować dwa kluczowe parametry dla każdej wykrytej butelki:
1.  **Obecność nakrętki:** Weryfikacja, czy butelka jest fabrycznie zamknięta.
2.  **Poziom napełnienia płynem:** Sprawdzenie, czy ilość płynu w butelce mieści się w akceptowalnej normie.

**Założenia środowiskowe i techniczne:**
* **Akwizycja obrazu:** Nieruchoma kamera (np. smartfon/kamera internetowa) ustawiona prostopadle do kierunku ruchu butelek.
* **Oświetlenie:** W miarę stabilne i równomierne, minimalizujące ostre odblaski na plastiku/szkle.
* **Tło:** Jednolite (np. jasny brystol), co znacząco ułatwi segmentację obiektów metodami klasycznymi.
* **Obiekty:** Butelki tego samego typu (np. przezroczyste z barwionym płynem), aby zachować stałą geometrię dla obszarów zainteresowania (ROI).

#### 2. Przegląd istniejących rozwiązań (AOI - Automated Optical Inspection)
W przemyśle rozlewniczym i farmaceutycznym problem ten rozwiązywany jest na dwa główne sposoby:

* **Klasyczne systemy wizyjne (Smart Cameras / Algorytmy deterministyczne):**
    * *Mocne strony:* Bardzo wysoka szybkość działania (tysiące sztuk na minutę), przewidywalność, mniejsze wymagania sprzętowe, łatwość kalibracji geometrii.
    * *Słabe strony:* Wysoka wrażliwość na zmienne warunki oświetleniowe oraz konieczność bardzo precyzyjnego pozycjonowania obiektów na taśmie.
* **Systemy oparte na Głębokim Uczeniu (Deep Learning np. YOLO, Mask R-CNN):**
    * *Mocne strony:* Ogromna odporność na zmiany oświetlenia, odblaski, rotację obiektów czy szum w tle. Nie wymagają ręcznego definiowania cech.
    * *Słabe strony:* Znacznie wyższe zapotrzebowanie na moc obliczeniową, "czarna skrzynka" (trudność w interpretacji błędów), konieczność zebrania i opisania (anotacji) potężnego zbioru danych.

**Wybór technologiczny:** W naszym projekcie decydujemy się na wykorzystanie systemu typu end-to-end opartego na głębokich sieciach neuronowych (np. architektura z rodziny YOLO). Zgodnie z wytycznymi przedmiotu, nasz wkład autorski zostanie zrealizowany w dwóch kluczowych obszarach. Po pierwsze, poprzez samodzielną akwizycję i manualną anotację (etykietowanie) dedykowanego zbioru statycznych zdjęć obrazujących defekty butelek. Po drugie, poprzez przeprowadzenie pogłębionej, krytycznej analizy rezultatów działania wytrenowanego modelu. W raporcie końcowym skupimy się na ewaluacji metryk skuteczności (m.in. Precision, Recall, mAP) oraz szczegółowej analizie przypadków błędnej klasyfikacji (Macierz Pomyłek - Confusion Matrix), co pozwoli na rzetelną ocenę ograniczeń wybranej sieci w zadaniach przemysłowej inspekcji optycznej.



#### 3. Zbiór danych (Dataset)
Aby zapewnić autorski charakter rozwiązania oraz mieć pełną kontrolę nad środowiskiem testowym, zbiór danych zostanie wygenerowany samodzielnie.
* **Forma danych:** Zdjęcia w formacie .jpg przedstawiające zasymulowany ruch taśmociągu z butelkami.
* **Zróżnicowanie próbek:** Zbiór będzie obejmował cztery główne scenariusze (klasy testowe):
    1.  *PASS:* Butelka pełna, z nakrętką (wzorzec).
    2.  *FAIL_1:* Brak nakrętki, poprawny poziom płynu.
    3.  *FAIL_2:* Za mało płynu, nakrętka obecna.
    4.  *FAIL_3:* Za mało płynu i brak nakrętki.

#### 4. Projekt techniczny rozwiązania (Pipeline)

Ze względu na wybór architektury typu *end-to-end* (rodzina YOLO), struktura systemu opiera się na przepływie danych przez głęboką sieć neuronową. W odróżnieniu od metod klasycznych, proces ekstrakcji cech odbywa się wewnątrz modelu. Poniżej przedstawiamy schemat docelowego potoku przetwarzania (tzw. *Inference Pipeline*) dla pojedynczego zdjęcia, wskazując główne bloki obliczeniowe i przekazywane dane:

**1. Moduł Akwizycji i Wczytywania Danych (Data Ingestion)**
* **Działanie:** System wsadowo (batch processing) pobiera statyczne obrazy z wcześniej przygotowanego katalogu testowego na dysku.
* **Dane wejściowe:** Plik graficzny (np. .jpg, .png).
* **Dane wyjściowe (przekazywane dalej):** Surowa macierz pikseli (obraz w przestrzeni RGB).

**2. Moduł Pre-processingu (Przygotowanie dla Sieci)**
* **Działanie:** Dostosowanie surowego obrazu do wymogów wejściowych sieci neuronowej. Obraz jest skalowany (np. do rozdzielczości 640x640 pikseli z zachowaniem proporcji - *letterboxing*) oraz poddawany normalizacji (wartości pikseli z zakresu 0-255 są rzutowane na zakres 0.0 - 1.0).
* **Algorytmy:** Interpolacja dwuliniowa (skalowanie), operacje macierzowe.
* **Dane wyjściowe:** Znormalizowany tensor wielowymiarowy (reprezentacja matematyczna obrazu gotowa do wejścia w sieć).

**3. Rdzeń Obliczeniowy – Inferencja Modelu (YOLO)**
* **Działanie:** Przekazanie tensora przez ukryte warstwy wytrenowanej, głębokiej sieci neuronowej. Sieć "end-to-end" jednocześnie dokonuje ekstrakcji cech (tzw. *backbone*) i predykcji lokalizacji oraz klas obiektów (tzw. *head*).
* **Algorytmy:** Splotowe sieci neuronowe (CNN), funkcje aktywacji, propagacja w przód (Forward Pass).
* **Dane wyjściowe:** Surowy wektor predykcji. Zawiera on dziesiątki tysięcy potencjalnych dopasowań, z których każde składa się z: współrzędnych *Bounding Boxa* (x_center, y_center, width, height), pewności detekcji (Confidence Score) oraz prawdopodobieństw przynależności do zdefiniowanych klas (np. `bottle_ok`, `missing_cap`, `low_liquid`).

**4. Moduł Post-processingu (Filtrowanie Wyników)**
* **Działanie:** Oczyszczenie surowych wyników z sieci. Odrzucane są detekcje o zbyt niskiej pewności, a powielone ramki dla tego samego obiektu są redukowane do jednej, najbardziej trafnej.
* **Algorytmy:** Progowanie ufności (Confidence Thresholding) oraz NMS (*Non-Maximum Suppression* - tłumienie wartości niemaksymalnych).
* **Dane wyjściowe:** Ostateczna, przefiltrowana lista wykrytych obiektów na zdjęciu wraz z ich etykietami i współrzędnymi.

**5. Moduł Agregacji, Wizualizacji i Oceny**
* **Działanie:** Nałożenie wyników na oryginalny obraz (narysowanie kolorowych ramek i etykiet). Dodatkowo, w trybie testowym, system porównuje predykcje z naszymi ręcznymi anotacjami (tzw. *Ground Truth*), aby wyliczyć statystyki błędów.
* **Algorytmy obliczeniowe (Wkład autorski):** Generowanie Macierzy Pomyłek (*Confusion Matrix*), obliczanie metryk: *Precision*, *Recall*, *mAP* (mean Average Precision).
* **Dane wyjściowe:** Zapisany plik graficzny z detekcjami oraz wygenerowane raporty statystyczne i wykresy skuteczności modelu dla każdej z klas defektów.


## ETAP 2: Prototyp rozwiązania
*(Do uzupełnienia do 6 maja)*
* [ ] Kod implementujący wczytywanie wideo.
* [ ] Działający pre-processing i prosta segmentacja.

## ETAP 3: Wyniki, testy i raport końcowy
*(Do uzupełnienia do 10 czerwca)*
* [ ] Opis działania zaimplementowanego systemu na podstawie gotowego kodu.
* [ ] Wyniki testów skuteczności na nagranym zbiorze danych.
* [ ] Analiza statystyczna (False Positives, False Negatives).
* [ ] Krytyczna analiza wyników i wnioski.