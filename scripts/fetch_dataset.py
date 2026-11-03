"""Download the public UCI diabetes readmission dataset."""
from pathlib import Path
import json


def main():
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -e .") from exc
    dataset = fetch_ucirepo(id=296)
    frame = dataset.data.original
    destination = Path("data/raw/diabetes_130_us_hospitals.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    metadata = {"source": "UCI Machine Learning Repository", "dataset_id": 296,
                "dataset_name": "Diabetes 130-US Hospitals for Years 1999-2008",
                "citation_doi": "10.24432/C5230J", "rows": len(frame), "columns": len(frame.columns)}
    destination.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(frame):,} rows to {destination}")

if __name__ == "__main__":
    main()
