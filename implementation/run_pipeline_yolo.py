"""
Pipeline YOLO

Kroki:
1. Dataset split
2. Dataset check
3. Training YOLO
4. Prediction YOLO

Uruchomienie:

python run_pipeline_yolo.py

lub:

python run_pipeline_yolo.py --step split
python run_pipeline_yolo.py --step check
python run_pipeline_yolo.py --step train
python run_pipeline_yolo.py --step predict
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from utils.dataset_split import main as split_data
from utils.check_datasets import main as check_data
from yolo.yolo_train import main as train
from yolo.yolo_predict import main as predict


def run_all():
    print("\n[1/4] Dataset split...")
    split_data()

    print("\n[2/4] Dataset check...")
    check_data()

    print("\n[3/4] Training YOLO...")
    train()

    print("\n[4/4] YOLO prediction...")
    predict()

    print("\n✅ YOLO pipeline zakończony!")


def main():
    parser = argparse.ArgumentParser(description="YOLO pipeline")

    parser.add_argument(
        "--step",
        choices=["all", "split", "check", "train", "predict"],
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

    else:
        run_all()


if __name__ == "__main__":
    main()