import csv
from pathlib import Path
from typing import Any

from weld_cv.inspect import inspect_file

CSV_COLUMNS = [
    "path",
    "expected_label",
    "predicted_label",
    "reason",
    "threshold",
    "roi_x",
    "roi_y",
    "roi_width",
    "roi_height",
    "roi_center_x",
    "roi_angle_deg",
    "terminal_x",
    "terminal_y",
    "terminal_width",
    "terminal_height",
    "weld_area",
    "fill_ratio",
    "weld_width",
    "weld_height",
    "vertical_coverage",
    "component_count",
]


def collect_images(raw_root: Path) -> list[Path]:
    return sorted(
        [path for path in raw_root.rglob("*") if path.is_file() and path.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}],
        key=lambda path: (path.name.lower(), str(path).lower()),
    )


def inspect_dataset(
    raw_root: Path,
    project_root: Path,
    config: dict[str, Any],
    debug_root: Path | None = None,
) -> list[dict[str, Any]]:
    return [inspect_file(path, project_root, config, debug_root=debug_root) for path in collect_images(raw_root)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
