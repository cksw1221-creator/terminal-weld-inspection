import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    width: int
    height: int
    center_x: int
    geometry_center_x: int
    terminal_bbox: tuple[int, int, int, int]
    angle_deg: float
    affine_matrix: np.ndarray
    inverse_matrix: np.ndarray
    terminal_quad: np.ndarray
    lower_half_quad: np.ndarray
    work_quad: np.ndarray
    roi_quad: np.ndarray
    line_quad: np.ndarray
    geometry_line_quad: np.ndarray

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


def locate_weld_roi(
    gray: np.ndarray,
    width: int = 320,
    height: int = 900,
    dark_threshold: int = 100,
    bright_threshold: int = 180,
    search_left_ratio: float = 0.25,
    search_right_ratio: float = 0.75,
    search_top_ratio: float = 0.20,
    center_search_ratio: float = 0.16,
    score_height_ratio: float = 0.60,
    bottom_margin: int = 0,
) -> Roi:
    contour = locate_terminal_contour(gray, dark_threshold=dark_threshold)
    bbox = cv2.boundingRect(contour) if contour is not None else (0, 0, gray.shape[1], gray.shape[0])
    coarse_rect = cv2.minAreaRect(contour) if contour is not None else _full_image_rect(gray)
    coarse_delta = _deskew_delta(coarse_rect)
    coarse_center = tuple(float(v) for v in coarse_rect[0])
    coarse_affine = cv2.getRotationMatrix2D(coarse_center, coarse_delta, 1.0)
    coarse_inverse = cv2.invertAffineTransform(coarse_affine)

    coarse_aligned = cv2.warpAffine(
        gray,
        coarse_affine,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    terminal_quad = cv2.boxPoints(coarse_rect).astype(np.float32)
    aligned_terminal_quad = _transform_points(terminal_quad, coarse_affine)
    ax, ay, aw, ah = cv2.boundingRect(aligned_terminal_quad.astype(np.float32))
    ax = _clamp(ax, 0, gray.shape[1] - 1)
    ay = _clamp(ay, 0, gray.shape[0] - 1)
    aw = min(aw, gray.shape[1] - ax)
    ah = min(ah, gray.shape[0] - ay)
    lower_half_y = ay + ah // 2
    lower_half_h = max(1, ay + ah - lower_half_y)
    lower_half_quad_aligned = np.array(
        [
            [ax, lower_half_y],
            [ax + aw - 1, lower_half_y],
            [ax + aw - 1, lower_half_y + lower_half_h - 1],
            [ax, lower_half_y + lower_half_h - 1],
        ],
        dtype=np.float32,
    )
    lower_half_quad = _transform_points(lower_half_quad_aligned, coarse_inverse)

    residual_delta = 0.0
    residual_affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    residual_delta = _estimate_lower_half_residual(
        coarse_aligned,
        lower_half_bbox=(ax, lower_half_y, aw, lower_half_h),
    )
    if abs(residual_delta) <= 35.0:
        lower_half_center = (float(ax + aw / 2.0), float(lower_half_y + lower_half_h / 2.0))
        residual_affine = cv2.getRotationMatrix2D(lower_half_center, residual_delta, 1.0)
    else:
        residual_delta = 0.0

    affine = _compose_affines(residual_affine, coarse_affine)
    inverse = cv2.invertAffineTransform(affine)
    angle_delta = coarse_delta + residual_delta

    aligned = cv2.warpAffine(
        gray,
        affine,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    aligned_terminal_quad = _transform_points(terminal_quad, affine)
    ax, ay, aw, ah = cv2.boundingRect(aligned_terminal_quad.astype(np.float32))
    ax = _clamp(ax, 0, gray.shape[1] - 1)
    ay = _clamp(ay, 0, gray.shape[0] - 1)
    aw = min(aw, gray.shape[1] - ax)
    ah = min(ah, gray.shape[0] - ay)

    bx, by, bw, bh = _locate_lower_work_bbox(
        aligned,
        dark_threshold=dark_threshold,
        coarse_bbox=(ax, ay, aw, ah),
    )

    geometry_center_x = bx + bw // 2
    if center_search_ratio > 0:
        half_search_width = max(1, int(round(bw * center_search_ratio)))
        search_left = geometry_center_x - half_search_width
        search_right = geometry_center_x + half_search_width
    else:
        search_left = bx + int(bw * search_left_ratio)
        search_right = bx + int(bw * search_right_ratio)
    search_left = _clamp(search_left, bx, bx + bw - 1)
    search_right = _clamp(search_right, search_left + 1, bx + bw)
    search_top = by + int(bh * search_top_ratio)
    search_bottom = by + bh
    search = aligned[search_top:search_bottom, search_left:search_right]

    if search.size == 0:
        center_x = geometry_center_x
    else:
        center_x = search_left + _best_vertical_weld_column(
            search,
            bright_threshold=bright_threshold,
            score_height_ratio=score_height_ratio,
        )

    x = _clamp(center_x - width // 2, 0, gray.shape[1] - width)
    work_bottom_y = by + bh - 1
    y = _clamp(work_bottom_y - bottom_margin - height + 1, 0, gray.shape[0] - height)
    width = min(width, gray.shape[1] - x)
    base_height = min(height, gray.shape[0] - y)
    height = min(int(round(base_height * 1.10)), gray.shape[0] - y)
    work_quad_aligned = np.array(
        [
            [bx, by],
            [bx + bw - 1, by],
            [bx + bw - 1, by + bh - 1],
            [bx, by + bh - 1],
        ],
        dtype=np.float32,
    )
    roi_quad_aligned = np.array(
        [
            [x, y],
            [x + width - 1, y],
            [x + width - 1, y + height - 1],
            [x, y + height - 1],
        ],
        dtype=np.float32,
    )
    line_quad_aligned = np.array(
        [
            [center_x, y],
            [center_x, y + height - 1],
        ],
        dtype=np.float32,
    )
    geometry_line_quad_aligned = np.array(
        [
            [geometry_center_x, y],
            [geometry_center_x, y + height - 1],
        ],
        dtype=np.float32,
    )
    return Roi(
        x=x,
        y=y,
        width=width,
        height=height,
        center_x=center_x,
        geometry_center_x=geometry_center_x,
        terminal_bbox=bbox,
        angle_deg=angle_delta,
        affine_matrix=affine,
        inverse_matrix=inverse,
        terminal_quad=terminal_quad,
        lower_half_quad=lower_half_quad,
        work_quad=_transform_points(work_quad_aligned, inverse),
        roi_quad=_transform_points(roi_quad_aligned, inverse),
        line_quad=_transform_points(line_quad_aligned, inverse),
        geometry_line_quad=_transform_points(geometry_line_quad_aligned, inverse),
    )


def locate_terminal_bbox(gray: np.ndarray, dark_threshold: int = 140) -> tuple[int, int, int, int]:
    contour = locate_terminal_contour(gray, dark_threshold=dark_threshold)
    if contour is None:
        return 0, 0, gray.shape[1], gray.shape[0]
    return cv2.boundingRect(contour)


def locate_terminal_contour(gray: np.ndarray, dark_threshold: int = 100):
    mask = cv2.inRange(gray, 0, dark_threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = gray.shape[0] * gray.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.02:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        touches_left = x <= 2
        touches_right = x + w >= gray.shape[1] - 2
        border_penalty = 0.45 if touches_left or touches_right else 1.0
        aspect_bonus = 1.3 if h >= w else 1.0
        candidates.append((area * border_penalty * aspect_bonus, contour))

    if not candidates:
        return max(contours, key=cv2.contourArea)
    return max(candidates, key=lambda item: item[0])[1]


def draw_roi(gray: np.ndarray, roi: Roi) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.polylines(canvas, [np.round(roi.work_quad).astype(np.int32)], True, (255, 255, 0), 3)
    cv2.polylines(canvas, [np.round(roi.roi_quad).astype(np.int32)], True, (0, 0, 255), 4)
    geometry_line = np.round(roi.geometry_line_quad).astype(np.int32)
    cv2.line(canvas, tuple(geometry_line[0]), tuple(geometry_line[1]), (0, 255, 255), 5)
    line = np.round(roi.line_quad).astype(np.int32)
    cv2.line(canvas, tuple(line[0]), tuple(line[1]), (255, 0, 0), 2)
    return canvas


def crop_roi(gray: np.ndarray, roi: Roi) -> np.ndarray:
    aligned = cv2.warpAffine(
        gray,
        roi.affine_matrix,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width].copy()


def _best_vertical_weld_column(search: np.ndarray, bright_threshold: int, score_height_ratio: float = 0.60) -> int:
    search_width = search.shape[1]
    geometry_center = search_width // 2
    score_height = _clamp(int(round(search.shape[0] * score_height_ratio)), 1, search.shape[0])
    region = search[:score_height, :].astype(np.float32)

    if region.size == 0 or search_width < 9:
        return geometry_center

    edge = np.abs(cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3))
    score = np.zeros(search_width, dtype=np.float32)
    dark_values = np.zeros(search_width, dtype=np.float32)
    bright_values = np.zeros(search_width, dtype=np.float32)
    edge_values = np.zeros(search_width, dtype=np.float32)
    widths = _candidate_core_widths(search_width)

    for x in range(search_width):
        geometry_score = 1.0 - min(1.0, abs(x - geometry_center) / max(1.0, search_width / 2.0))
        dark_score, bright_score, edge_score = _best_window_scores(region, edge, x, widths)
        dark_values[x] = dark_score
        bright_values[x] = bright_score
        edge_values[x] = edge_score
        score[x] = 0.60 * geometry_score + 0.30 * dark_score + 0.08 * edge_score + 0.02 * bright_score

    if float(dark_values.max(initial=0.0)) < 0.08 and float(edge_values.max(initial=0.0)) < 0.08:
        return geometry_center

    if score.size >= 7:
        kernel = np.ones(7, dtype=np.float32) / 7.0
        score = np.convolve(score, kernel, mode="same")

    return int(score.argmax())


def _estimate_lower_half_residual(
    aligned: np.ndarray,
    lower_half_bbox: tuple[int, int, int, int],
) -> float:
    x, y, width, height = lower_half_bbox
    x = _clamp(x, 0, aligned.shape[1] - 1)
    y = _clamp(y, 0, aligned.shape[0] - 1)
    width = min(width, aligned.shape[1] - x)
    height = min(height, aligned.shape[0] - y)
    if width <= 1 or height <= 1:
        return 0.0

    crop = aligned[y : y + height, x : x + width]
    mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=30,
        minLineLength=max(30, int(height * 0.35)),
        maxLineGap=10,
    )
    if lines is None:
        return 0.0

    deltas: list[tuple[float, float]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(abs(angle) - 90.0) > 30.0:
            continue
        target = 90.0 if angle >= 0.0 else -90.0
        delta = angle - target
        length = math.hypot(x2 - x1, y2 - y1)
        deltas.append((delta, length))

    if not deltas:
        return 0.0
    total_weight = sum(weight for _delta, weight in deltas)
    if total_weight <= 0:
        return 0.0
    return float(sum(delta * weight for delta, weight in deltas) / total_weight)


def _locate_lower_work_bbox(
    aligned: np.ndarray,
    dark_threshold: int,
    coarse_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    coarse_x, coarse_y, coarse_w, coarse_h = coarse_bbox
    coarse_x = _clamp(coarse_x, 0, aligned.shape[1] - 1)
    coarse_y = _clamp(coarse_y, 0, aligned.shape[0] - 1)
    coarse_w = min(coarse_w, aligned.shape[1] - coarse_x)
    coarse_h = min(coarse_h, aligned.shape[0] - coarse_y)
    if coarse_w <= 1 or coarse_h <= 1:
        return coarse_x, coarse_y, coarse_w, coarse_h

    mask = cv2.inRange(aligned, 0, dark_threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    crop = mask[coarse_y : coarse_y + coarse_h, coarse_x : coarse_x + coarse_w]

    row_counts = np.count_nonzero(crop, axis=1)
    valid_rows = np.flatnonzero(row_counts >= max(8, int(coarse_w * 0.08)))
    if valid_rows.size == 0:
        return coarse_x, coarse_y, coarse_w, coarse_h

    bottom_rel = int(valid_rows[-1])
    band_top_rel = max(0, bottom_rel - max(30, int(coarse_h * 0.45)))
    lower_band = crop[band_top_rel : bottom_rel + 1, :]
    min_row_width = max(8, int(coarse_w * 0.08))
    left_edges: list[int] = []
    right_edges: list[int] = []
    for row in lower_band:
        xs = np.flatnonzero(row > 0)
        if xs.size >= min_row_width:
            left_edges.append(int(xs[0]))
            right_edges.append(int(xs[-1]))

    if not left_edges or not right_edges:
        return coarse_x, coarse_y, coarse_w, coarse_h

    left_rel = int(round(float(np.median(left_edges))))
    right_rel = int(round(float(np.median(right_edges))))
    if right_rel <= left_rel:
        return coarse_x, coarse_y, coarse_w, coarse_h

    work_width = right_rel - left_rel + 1
    min_overlap = max(6, int(work_width * 0.55))
    work_rows: list[int] = []
    for row_rel in range(0, bottom_rel + 1):
        overlap = int(np.count_nonzero(crop[row_rel, left_rel : right_rel + 1]))
        if overlap >= min_overlap:
            work_rows.append(row_rel)

    lower_limit = max(0, bottom_rel - max(40, int(coarse_h * 0.65)))
    lower_work_rows = [row for row in work_rows if row >= lower_limit]
    top_rel = min(lower_work_rows) if lower_work_rows else band_top_rel

    x = coarse_x + left_rel
    y = coarse_y + top_rel
    width = right_rel - left_rel + 1
    height = bottom_rel - top_rel + 1
    return x, y, width, height


def _candidate_core_widths(search_width: int) -> list[int]:
    widths = [12, 20, 32, 48]
    max_width = max(3, search_width // 3)
    return [width for width in widths if width <= max_width] or [max(3, max_width)]


def _best_window_scores(
    region: np.ndarray,
    edge: np.ndarray,
    center_x: int,
    widths: list[int],
) -> tuple[float, float, float]:
    best_dark = 0.0
    best_bright = 0.0
    best_edge = 0.0
    for width in widths:
        half = width // 2
        core_left = center_x - half
        core_right = core_left + width
        guard = max(4, width // 4)
        side_width = max(8, int(round(width * 1.75)))
        left_left = core_left - guard - side_width
        left_right = core_left - guard
        right_left = core_right + guard
        right_right = right_left + side_width

        if left_left < 0 or right_right > region.shape[1] or core_left < 0 or core_right > region.shape[1]:
            continue

        core = region[:, core_left:core_right]
        left = region[:, left_left:left_right]
        right = region[:, right_left:right_right]
        core_level = float(np.median(core))
        left_level = float(np.median(left))
        right_level = float(np.median(right))

        dark = min(left_level - core_level, right_level - core_level)
        bright = min(core_level - left_level, core_level - right_level)
        if dark > 0:
            dark_continuity = _contrast_continuity(core, left, right, -1)
            dark_score = min(1.0, dark / 80.0) * (0.55 + 0.45 * dark_continuity)
            best_dark = max(best_dark, dark_score)
        if bright > 0:
            bright_continuity = _contrast_continuity(core, left, right, 1)
            bright_score = min(1.0, bright / 80.0) * (0.55 + 0.45 * bright_continuity)
            best_bright = max(best_bright, bright_score)

        left_edge = _edge_band_mean(edge, core_left)
        right_edge = _edge_band_mean(edge, core_right)
        edge_score = min(1.0, min(left_edge, right_edge) / 180.0)

        best_edge = max(best_edge, edge_score)

    return best_dark, best_bright, best_edge


def _contrast_continuity(core: np.ndarray, left: np.ndarray, right: np.ndarray, polarity: int) -> float:
    segment_count = min(10, max(1, core.shape[0] // 12))
    if segment_count <= 1:
        return 1.0
    hits = 0
    for segment in np.array_split(np.arange(core.shape[0]), segment_count):
        core_level = float(np.median(core[segment, :]))
        left_level = float(np.median(left[segment, :]))
        right_level = float(np.median(right[segment, :]))
        if polarity > 0:
            contrast = min(core_level - left_level, core_level - right_level)
        else:
            contrast = min(left_level - core_level, right_level - core_level)
        if contrast >= 8.0:
            hits += 1
    return hits / segment_count


def _edge_band_mean(edge: np.ndarray, x: int) -> float:
    left = _clamp(x - 2, 0, edge.shape[1])
    right = _clamp(x + 3, left + 1, edge.shape[1])
    return float(edge[:, left:right].mean())


def _clamp(value: int, low: int, high: int) -> int:
    if high < low:
        return low
    return max(low, min(int(value), high))


def _deskew_delta(rect) -> float:
    box = cv2.boxPoints(rect).astype(np.float32)
    edges = [box[(index + 1) % 4] - box[index] for index in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    long_edge = edges[int(np.argmax(lengths))]
    long_angle = float(np.degrees(np.arctan2(long_edge[1], long_edge[0])))
    while long_angle <= -90.0:
        long_angle += 180.0
    while long_angle > 90.0:
        long_angle -= 180.0
    target = 90.0 if long_angle >= 0 else -90.0
    delta = long_angle - target
    if delta > 90.0:
        delta -= 180.0
    if delta < -90.0:
        delta += 180.0
    return float(delta)


def _compose_affines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_3x3 = np.vstack([first, [0.0, 0.0, 1.0]])
    second_3x3 = np.vstack([second, [0.0, 0.0, 1.0]])
    combined = first_3x3 @ second_3x3
    return combined[:2, :].astype(np.float32)


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(points.reshape(1, -1, 2), matrix).reshape(-1, 2)


def _full_image_rect(gray: np.ndarray):
    return ((gray.shape[1] / 2.0, gray.shape[0] / 2.0), (float(gray.shape[1]), float(gray.shape[0])), 0.0)
