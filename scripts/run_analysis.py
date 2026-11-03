"""Run CareReturn training and held-out evaluation."""
import argparse
import json
from care_return.data import read_encounters
from care_return.pipeline import train_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/diabetes_130_us_hospitals.csv", help="Path to the UCI CSV")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    report = train_project(read_encounters(args.input), seed=args.seed)
    print(json.dumps({"selected_model": report["selected_model"], "held_out_test": report["held_out_test"]}, indent=2))

if __name__ == "__main__":
    main()
