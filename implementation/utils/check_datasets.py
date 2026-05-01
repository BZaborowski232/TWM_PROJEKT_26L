from pathlib import Path
from collections import Counter


#ten plik sprawdza, ile klas ma dataset zeby je poprawenie zdefiniować w data.yaml, a także ile jest pustych plików etykiet, które mogą powodować błędy podczas treningu


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABEL_DIRS = [
    PROJECT_ROOT / "labels" / "train",
    PROJECT_ROOT / "labels" / "val",
]


def main():
    class_counter = Counter()
    files_count = 0
    empty_files = 0

    for label_dir in LABEL_DIRS:
        if not label_dir.exists():
            print(f"Brak folderu: {label_dir}")
            continue

        for txt_path in label_dir.glob("*.txt"):
            files_count += 1
            content = txt_path.read_text(encoding="utf-8").strip()

            if not content:
                empty_files += 1
                continue

            for line in content.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = parts[0]
                class_counter[class_id] += 1

    print("Podsumowanie etykiet YOLO")
    print("========================")
    print(f"Liczba plików etykiet: {files_count}")
    print(f"Puste pliki etykiet:   {empty_files}")
    print()
    print("Wykryte klasy:")

    for class_id, count in sorted(class_counter.items(), key=lambda x: int(x[0])):
        print(f"klasa {class_id}: {count} obiektów")


if __name__ == "__main__":
    main()