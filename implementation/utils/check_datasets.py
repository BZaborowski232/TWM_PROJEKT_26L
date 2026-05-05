from pathlib import Path
from collections import Counter


#ten plik sprawdza, ile klas ma dataset zeby je poprawenie zdefiniować w data.yaml, 
# a także ile jest pustych plików etykiet, które mogą powodować błędy podczas treningu


# ścieżka do głównego katalogu projektu
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# katalogi z etykietami (train i val)
LABEL_DIRS = [
    PROJECT_ROOT / "labels" / "train",
    PROJECT_ROOT / "labels" / "val",
]


def main():
    class_counter = Counter()  # liczy wystąpienia klas
    files_count = 0            # liczba plików .txt
    empty_files = 0            # liczba pustych plików

    # przejście po katalogach z etykietami
    for label_dir in LABEL_DIRS:
        if not label_dir.exists():
            print(f"Brak folderu: {label_dir}")
            continue

        # iteracja po plikach YOLO (.txt)
        for txt_path in label_dir.glob("*.txt"):
            files_count += 1

            # wczytaj zawartość pliku
            content = txt_path.read_text(encoding="utf-8").strip()

            # jeśli pusty → policz i pomiń
            if not content:
                empty_files += 1
                continue

            # każda linia = jeden obiekt
            for line in content.splitlines():
                parts = line.strip().split()

                # zabezpieczenie przed błędną linią
                if not parts:
                    continue

                # pierwsza wartość = class_id
                class_id = parts[0]

                # zwiększ licznik dla tej klasy
                class_counter[class_id] += 1

    # 🔷 podsumowanie
    print("Podsumowanie etykiet YOLO:")
    print()

    print(f"Liczba plików etykiet: {files_count}")
    print(f"Puste pliki etykiet:   {empty_files}")
    print()

    print("Wykryte klasy:")

    # sortowanie klas (np. 0,1,2...)
    for class_id, count in sorted(class_counter.items(), key=lambda x: int(x[0])):
        print(f"klasa {class_id}: {count} obiektów")


# uruchomienie skryptu
if __name__ == "__main__":
    main()