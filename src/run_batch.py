import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / ".vendor"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from weld_cv.batch import inspect_dataset, write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenCV terminal weld inspection.")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--config", default="config/inspection.json", help="JSON config")
    parser.add_argument("--raw", default="data/raw", help="Raw image directory")
    parser.add_argument("--out", default="results/report.csv", help="CSV report path")
    parser.add_argument("--debug", default="results/debug", help="Debug image output directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    with (root / args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    rows = inspect_dataset(
        raw_root=(root / args.raw).resolve(),
        project_root=root,
        config=config,
        debug_root=(root / args.debug).resolve(),
    )
    write_csv((root / args.out).resolve(), rows)
    print(f"inspected={len(rows)} report={(root / args.out).resolve()} debug={(root / args.debug).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

