# TWM - Techniki Widzenia Maszynowego - Projekt 26L
## Temat: "Optyczna inspekcja obiektów na taśmociągu"
### Zespół:
* Piotr Walczak 315220
* Katarzyna Wawer 311683
* Bartosz Zaborowski 319996

### ETAP 1 - Założenia wstepne projektu

#### 1. Opis problemu

**Cel projektu:** 
Celem projektu jest stworzenie systemu wizyjnego opartego na metodach klasycznego przetwarzania obrazu, którego zadaniem jest automatyczna inspekcja jakości butelek przemieszczających się na symulowanym taśmociągu. System ma za zadanie w czasie rzeczywistym weryfikować kluczowe parametry dla każdej wykrytej butelki:

1.  **Obecność nakrętki:** Weryfikacja, czy butelka jest fabrycznie zamknięta.
2.  **Poziom napełnienia płynem:** Sprawdzenie, czy ilość płynu w butelce mieści się w akceptowalnej normie.
3. **Stan płynu w butelce:** Czy nie doszło do zanieczyszczenia

**Założenia środowiskowe i techniczne:**
* **Akwizycja obrazu:** Nieruchoma kamera (np. smartfon/kamera internetowa) ustawiona prostopadle do kierunku ruchu butelek.
* **Oświetlenie:** W miarę stabilne i równomierne, minimalizujące ostre odblaski na plastiku/szkle.
* **Tło:** Stałe dla wszystkich zdjęć
* **Obiekty:** Butelki tego samego typu, aby zachować stałą geometrię dla obszarów zainteresowania.

#### 2. Przegląd istniejących rozwiązań (AOI - Automated Optical Inspection)
W przemyśle rozlewniczym i farmaceutycznym problem ten rozwiązywany jest na dwa główne sposoby:

* **Klasyczne systemy wizyjne (Smart Cameras / Algorytmy deterministyczne):**
    * *Mocne strony:* Bardzo wysoka szybkość działania (tysiące sztuk na minutę), przewidywalność, mniejsze wymagania sprzętowe, łatwość kalibracji geometrii.
    * *Słabe strony:* Wysoka wrażliwość na zmienne warunki oświetleniowe oraz konieczność bardzo precyzyjnego pozycjonowania obiektów na taśmie.
* **Systemy oparte na Głębokim Uczeniu (Deep Learning np. YOLO, Mask R-CNN):**
    * *Mocne strony:* Ogromna odporność na zmiany oświetlenia, odblaski, rotację obiektów czy szum w tle. Nie wymagają ręcznego definiowania cech.
    * *Słabe strony:* Znacznie wyższe zapotrzebowanie na moc obliczeniową, "czarna skrzynka" (trudność w interpretacji błędów), konieczność zebrania i opisania potężnego zbioru danych.

**Wybór technologiczny:** W naszym projekcie decydujemy się na wykorzystanie systemu typu end-to-end opartego na głębokich sieciach neuronowych (np. architektura z rodziny YOLO). Planujemy przeprowadzenie pogłębionej, krytycznej analizy rezultatów działania wytrenowanego modelu. W raporcie końcowym skupimy się na ewaluacji metryk skuteczności (m.in. Precision, Recall, mAP) oraz szczegółowej analizie przypadków błędnej klasyfikacji (Macierz Pomyłek), co pozwoli na rzetelną ocenę ograniczeń wybranej sieci w zadaniach przemysłowej inspekcji optycznej.


### 3. Zbiór danych (Dataset)
Projekt opiera się na ogólnodostępnym, specjalistycznym zbiorze obrazów z platformy Kaggle: *"Water Bottle Defect-Level Detection Dataset"*. Zbiór ten został stworzony z myślą o trenowaniu modeli detekcji obiektów i dobrze symuluje warunki przemysłowej inspekcji optycznej. Zbiór składa się ze statycznych zdjęć przedstawiających butelki z wodą, kategoryzowanych pod kątem poprawności napełnienia płynem, jego jakości oraz obecności nakrętki. Obrazy uwzględniają różne warunki oświetleniowe i ujęcia, co sprzyja uogólnieniu (odporności) modelu.

* **Format anotacji (Etykiety YOLO):** Każdemu zdjęciu z folderu `images` odpowiada plik tekstowy w folderze `labels`. Anotacje przygotowane są w standardzie architektur YOLO. Każda linia w pliku `.txt` opisuje jeden wykryty obiekt (np. brakującą nakrętkę) w formacie: `<Klasa_ID> <Środek_X> <Środek_Y> <Szerokość> <Wysokość>`. Wartości te są znormalizowane (zapisane jako ułamki względem wymiarów zdjęcia), co pozwala sieci na niezależne od rozdzielczości przetwarzanie geometrii obiektów (tzw. *letterboxing*).
* **Struktura i podział danych:** Oryginalny zbiór pobrany z platformy dostarcza dane podzielone na dwie grupy. W celu zapewnienia rzetelnej metodologii badawczej, struktura ta zostanie przez nas odpowiednio zmodyfikowana do postaci trzech niezależnych zbiorów:
    * **`train` (Zbiór treningowy):** Baza ucząca (zdjęcia oraz dołączone do nich etykiety), na której sieć w sposób iteracyjny optymalizuje swoje wagi.
    * **`val` (Zbiór walidacyjny):** Wydzielona paczka danych używana wewnętrznie przez algorytm w trakcie procesu uczenia (po każdej tzw. epoce). Rozwiązywanie tego zbioru pozwala monitorować metryki w czasie rzeczywistym i skutecznie zapobiegać zjawisku przeuczenia (*overfitting*).
    * **`test` (Zbiór testowy):** Ze względu na brak dedykowanego zbioru testowego w oryginalnych danych, dohierzemy reprezentatywną próbkę obrazów (wraz z ukrytymi dla modelu etykietami docelowymi) do osobnego katalogu. Zbiór ten zostanie całkowicie wyłączony z procesu treningu. Wykorzystamy go wyłącznie na samym końcu projektu do weryfikacji wyników wytrenowanego modelu. Pozwoli to na obiektywne zestawienie wyników sieci z rzeczywistością i wygenerowanie statystyk, w tym Macierzy Pomyłek.

Poniżej prezentujemy wybrane poglądowe zdjęcia z datasetu aby zobrazować jego przekrój:

* **Uszkodzona etykieta:**
![](images/train/damaged_label_0099_20260210_151750.jpg)

* **Zanieczyszczenia:**
![](images/train/debris_0107_20260210_150528.jpg)

* **Uszkodzona nakrętka:**
![](images/train/loose_cap_0046_20260210_152220.jpg)

* **Dobra butelka:**
![](images/train/good_0563_20260210_131754.jpg)


#### 4. Wstępny projekt techniczny rozwiązania (Pipeline)

Ze względu na wybór architektury typu *end-to-end* (rodzina YOLO), struktura systemu opiera się na przepływie danych przez głęboką sieć neuronową. W odróżnieniu od metod klasycznych, proces ekstrakcji cech odbywa się wewnątrz modelu. Poniżej przedstawiamy schemat docelowego potoku przetwarzania (tzw. *Inference Pipeline*) dla pojedynczego zdjęcia, wskazując główne bloki obliczeniowe i przekazywane dane:

**1. Moduł Akwizycji i Wczytywania Danych (Data Ingestion)**
* **Działanie:** System wsadowo (batch processing) pobiera statyczne obrazy z wcześniej przygotowanego katalogu testowego na dysku.
* **Dane wejściowe:** Plik graficzny (np. .jpg, .png).
* **Dane wyjściowe (przekazywane dalej):** Surowa macierz pikseli (obraz w przestrzeni RGB).

**2. Moduł Pre-processingu (Przygotowanie dla Sieci)**
* **Działanie:** Dostosowanie surowego obrazu do wymogów wejściowych sieci neuronowej. Obraz jest skalowany (np. do rozdzielczości 640x640 pikseli z zachowaniem proporcji - *letterboxing*) oraz poddawany normalizacji (wartości pikseli z zakresu 0-255 są rzutowane na zakres 0.0 - 1.0).
* **Algorytmy:** Interpolacja dwuliniowa (skalowanie), operacje macierzowe.
* **Dane wyjściowe:** Znormalizowany tensor wielowymiarowy (reprezentacja matematyczna obrazu gotowa do wejścia w sieć).

**3. Rdzeń Obliczeniowy**
* **Działanie:** Przekazanie tensora przez ukryte warstwy wytrenowanej, głębokiej sieci neuronowej. Sieć "end-to-end" jednocześnie dokonuje ekstrakcji cech i predykcji lokalizacji oraz klas obiektów.
* **Algorytmy:** Splotowe sieci neuronowe (CNN), funkcje aktywacji, propagacja w przód (Forward Pass).
* **Dane wyjściowe:** Surowy wektor predykcji. Zawiera on dziesiątki tysięcy potencjalnych dopasowań, z których każde składa się z: współrzędnych *Bounding Boxa* (x_center, y_center, width, height), pewności detekcji (Confidence Score) oraz prawdopodobieństw przynależności do zdefiniowanych klas (np. `bottle_ok`, `missing_cap`, `low_liquid`).

**4. Moduł Post-processingu (Filtrowanie Wyników)**
* **Działanie:** Oczyszczenie surowych wyników z sieci. Odrzucane są detekcje o zbyt niskiej pewności, a powielone ramki dla tego samego obiektu są redukowane do jednej, najbardziej trafnej.
* **Algorytmy:** Progowanie ufności (Confidence Thresholding) oraz NMS (*Non-Maximum Suppression* - tłumienie wartości niemaksymalnych).
* **Dane wyjściowe:** Ostateczna, przefiltrowana lista wykrytych obiektów na zdjęciu wraz z ich etykietami i współrzędnymi.

**5. Moduł Agregacji, Wizualizacji i Oceny**
* **Działanie:** Nałożenie wyników na oryginalny obraz (narysowanie kolorowych ramek i etykiet). Dodatkowo, w trybie testowym, system porównuje predykcje z naszymi ręcznymi anotacjami, aby wyliczyć statystyki błędów.
* **Algorytmy obliczeniowe (Wkład autorski):** Generowanie Macierzy Pomyłek (*Confusion Matrix*), obliczanie metryk: *Precision*, *Recall*, *mAP* (mean Average Precision).
* **Dane wyjściowe:** Zapisany plik graficzny z detekcjami oraz wygenerowane raporty statystyczne i wykresy skuteczności modelu dla każdej z klas defektów.

W celu zapewnienia obiektywnego punktu odniesienia dla wyników sieci YOLO, zaimplementowany zostanie dodatkowy, klasyczny moduł weryfikacji. Będzie on działał przykładowo w oparciu o statyczne wydzielenie obszaru zainteresowania (ROI), a detekcja defektów takich oprze się na prostej analizie cech pikseli, takich jak odchylenia w histogramie kolorów (np. w przestrzeni HSV). Takie podejście pozwoli na pporównaniu podejść i udowodnieniu, że dla specyficznych, prostych wizualnie defektów metody klasyczne mogą stanowić znacznie szybszą i bardziej zoptymalizowaną obliczeniowo alternatywę dla złożonych modeli typu end-to-end.

