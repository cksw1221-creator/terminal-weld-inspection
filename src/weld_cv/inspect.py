from pathlib import Path
from typing import Any

import cv2
import numpy as np

from weld_cv.image_io import read_gray, write_image
from weld_cv.labels import expected_label
from weld_cv.roi import crop_roi, draw_roi, locate_weld_roi


def inspect_file(path: Path, project_root: Path, config: dict[str, Any], debug_root: Path | None = None) -> dict[str, Any]:
    gray = read_gray(path)
    roi_config = config["roi"]
    roi = locate_weld_roi(gray, **roi_config)
    roi_image = crop_roi(gray, roi)
    threshold = int(config["threshold"]["bright_value"])
    features = extract_features(roi_image, threshold)
    prediction, reason = classify_features(features, config["rules"])

    if debug_root is not None:
        _write_debug_images(debug_root, path, gray, roi_image, draw_roi(gray, roi), prediction, features)

    relative = path.relative_to(project_root) if path.is_relative_to(project_root) else path
    row: dict[str, Any] = {
        "path": str(relative),
        "expected_label": expected_label(path),
        "predicted_label": prediction,
        "reason": reason,
        "threshold": threshold,
        "roi_x": roi.x,
        "roi_y": roi.y,
        "roi_width": roi.width,
        "roi_height": roi.height,
        "roi_center_x": roi.center_x,
        "roi_angle_deg": roi.angle_deg,
        "terminal_x": roi.terminal_bbox[0],
        "terminal_y": roi.terminal_bbox[1],
        "terminal_width": roi.terminal_bbox[2],
        "terminal_height": roi.terminal_bbox[3],
    }
    row.update(features)
    return row


def extract_features(roi_gray: np.ndarray, threshold: int) -> dict[str, float]:
    _, mask = cv2.threshold(roi_gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    area = int(cv2.countNonZero(mask))
    total = int(mask.shape[0] * mask.shape[1])
    if area == 0:
        return {
            "weld_area": 0,
            "fill_ratio": 0.0,
            "weld_width": 0,
            "weld_height": 0,
            "vertical_coverage": 0.0,
            "component_count": 0,
        }

    points = cv2.findNonZero(mask)
    x, y, width, height = cv2.boundingRect(points)
    component_count = _component_count(mask, min_area=20)
    return {
        "weld_area": area,
        "fill_ratio": area / total,
        "weld_width": width,
        "weld_height": height,
        "vertical_coverage": height / mask.shape[0],
        "component_count": component_count,
    }


def classify_features(features: dict[str, float], rules: dict[str, float]) -> tuple[str, str]:
    if features["weld_area"] < rules["missing_area_threshold"]:
        return "missing", "area_below_missing_threshold"
    if features["fill_ratio"] < rules["less_fill_threshold"]:
        return "less", "fill_ratio_below_less_threshold"
    if features["weld_height"] < rules["min_height_px"]:
        return "less", "height_below_min_height"
    return "ok", "rules_passed"


def _component_count(mask: np.ndarray, min_area: int) -> int:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep += 1
    return keep


def _write_debug_images(
    debug_root: Path,
    source_path: Path,
    gray: np.ndarray,
    roi_image: np.ndarray,
    overlay: np.ndarray,
    prediction: str,
    features: dict[str, float],
) -> None:
    safe_stem = source_path.stem.replace(" ", "_")
    write_image(debug_root / "overlay" / f"{safe_stem}__overlay.jpg", overlay)
    write_image(debug_root / "roi" / f"{safe_stem}__roi.bmp", roi_image)
    if prediction != "ok" or features["weld_area"] == 0:
        write_image(debug_root / "review" / prediction / f"{safe_stem}__overlay.jpg", overlay)
        write_image(debug_root / "review" / prediction / f"{safe_stem}__roi.bmp", roi_image)
