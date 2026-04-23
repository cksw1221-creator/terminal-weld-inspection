from pathlib import Path

import cv2
import numpy as np


def read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".bmp"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    encoded.tofile(str(path))

