import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".vendor"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import cv2

from weld_cv.inspect import classify_features, extract_features
from weld_cv.image_io import read_gray
from weld_cv.roi import _deskew_delta, crop_roi, draw_roi, locate_weld_roi


class VisionTests(unittest.TestCase):
    def test_locates_vertical_weld_line_inside_dark_terminal(self):
        image = np.full((180, 220), 180, dtype=np.uint8)
        image[20:165, 40:170] = 55
        image[55:155, 95:105] = 235

        roi = locate_weld_roi(image, width=50, height=120)

        self.assertLessEqual(abs(roi.center_x - 100), 8)
        self.assertEqual(roi.width, 50)
        self.assertEqual(roi.height, 132)
        self.assertGreaterEqual(roi.x, 0)
        self.assertGreaterEqual(roi.y, 0)

    def test_rotated_roi_follows_angled_terminal(self):
        image = np.full((220, 260), 180, dtype=np.uint8)
        rect = ((130, 110), (90, 170), 12)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(image, [box], 0, 55, -1)
        cv2.line(image, (113, 45), (148, 180), 235, 8)

        roi = locate_weld_roi(image, width=60, height=150, dark_threshold=100)
        cropped = crop_roi(image, roi)
        features = extract_features(cropped, 180)

        self.assertGreater(abs(roi.angle_deg), 5)
        self.assertGreater(features["weld_height"], 100)
        self.assertGreater(features["weld_area"], 600)

    def test_dark_gap_center_ignores_lower_bright_blob(self):
        image = np.full((240, 260), 180, dtype=np.uint8)
        image[20:220, 40:220] = 70
        image[55:170, 126:134] = 35
        image[160:210, 165:205] = 240

        roi = locate_weld_roi(
            image,
            width=60,
            height=170,
            dark_threshold=100,
            bright_threshold=180,
            search_left_ratio=0.25,
            search_right_ratio=0.75,
            search_top_ratio=0.2,
            bottom_margin=5,
        )

        self.assertLessEqual(abs(roi.center_x - 130), 8)

    def test_bright_seam_center_ignores_lower_bright_blob(self):
        image = np.full((240, 260), 180, dtype=np.uint8)
        image[20:220, 40:220] = 70
        image[55:145, 126:134] = 235
        image[135:220, 165:205] = 245

        roi = locate_weld_roi(
            image,
            width=60,
            height=170,
            dark_threshold=100,
            bright_threshold=180,
            search_left_ratio=0.25,
            search_right_ratio=0.75,
            search_top_ratio=0.2,
            bottom_margin=5,
        )

        self.assertLessEqual(abs(roi.center_x - 130), 8)

    def test_geometry_center_uses_lower_rectangular_work_area(self):
        image = np.full((260, 280), 180, dtype=np.uint8)
        cv2.ellipse(image, (120, 90), (85, 65), 0, 0, 360, 70, -1)
        image[105:235, 100:180] = 70

        roi = locate_weld_roi(
            image,
            width=70,
            height=120,
            dark_threshold=100,
            bright_threshold=180,
            bottom_margin=0,
        )

        self.assertLessEqual(abs(roi.geometry_center_x - 140), 14)
        self.assertLessEqual(abs(roi.center_x - 140), 14)

    def test_roi_bottom_aligns_to_lower_work_area_bottom(self):
        image = np.full((260, 280), 180, dtype=np.uint8)
        cv2.ellipse(image, (120, 90), (85, 65), 0, 0, 360, 70, -1)
        image[105:235, 100:180] = 70

        roi = locate_weld_roi(
            image,
            width=70,
            height=120,
            dark_threshold=100,
            bright_threshold=180,
            bottom_margin=0,
        )

        self.assertLessEqual(abs((roi.y + roi.height - 1) - 246), 2)

    def test_rotation_uses_lower_half_not_upper_round_head(self):
        image = np.full((320, 320), 180, dtype=np.uint8)
        lower = ((170, 220), (90, 150), 12)
        upper = ((125, 90), (170, 130), -18)
        cv2.drawContours(image, [cv2.boxPoints(lower).astype(np.int32)], 0, 70, -1)
        cv2.ellipse(image, upper, 70, -1)

        roi = locate_weld_roi(image, width=70, height=120, dark_threshold=100)

        self.assertLessEqual(abs(roi.angle_deg - 12.0), 2.0)

    def test_exposes_lower_half_angle_box_for_debug(self):
        image = np.full((220, 260), 180, dtype=np.uint8)
        image[20:200, 70:190] = 70

        roi = locate_weld_roi(image, width=60, height=120, dark_threshold=100)

        self.assertEqual(roi.lower_half_quad.shape, (4, 2))

    def test_overlay_hides_green_lower_half_box(self):
        image = np.full((220, 260), 180, dtype=np.uint8)
        image[20:200, 70:190] = 70

        roi = locate_weld_roi(image, width=60, height=120, dark_threshold=100)
        overlay = draw_roi(image, roi)

        pure_green = np.all(overlay == np.array([0, 255, 0], dtype=np.uint8), axis=2)
        self.assertEqual(int(np.count_nonzero(pure_green)), 0)

    def test_missing36_center_does_not_snap_to_bright_edge(self):
        image_path = ROOT / "data" / "raw" / "ng" / "missing (36).bmp"
        self.assertTrue(image_path.exists())
        image = read_gray(image_path)

        roi = locate_weld_roi(
            image,
            width=340,
            height=950,
            dark_threshold=100,
            bright_threshold=180,
            center_search_ratio=0.16,
            score_height_ratio=0.60,
            bottom_margin=0,
        )

        self.assertLessEqual(abs(roi.center_x - roi.geometry_center_x), 15)

    def test_roi_height_extends_downward_by_10_percent(self):
        image = np.full((260, 280), 180, dtype=np.uint8)
        cv2.ellipse(image, (120, 90), (85, 65), 0, 0, 360, 70, -1)
        image[105:235, 100:180] = 70

        roi = locate_weld_roi(
            image,
            width=70,
            height=120,
            dark_threshold=100,
            bright_threshold=180,
            bottom_margin=0,
        )

        self.assertEqual(roi.y, 115)
        self.assertEqual(roi.height, 132)
        self.assertEqual(roi.y + roi.height - 1, 246)

    def test_deskew_delta_uses_correct_opencv_rotation_sign(self):
        self.assertLess(_deskew_delta(((0, 0), (793, 1331), -22.37)), 0)
        self.assertGreater(_deskew_delta(((0, 0), (1173, 815), -86.13)), 0)

    def test_classifies_missing_less_and_ok_from_bright_weld_features(self):
        rules = {
            "missing_area_threshold": 100,
            "less_fill_threshold": 0.05,
            "min_height_px": 60,
        }

        missing = np.zeros((100, 50), dtype=np.uint8)
        less = np.zeros((100, 50), dtype=np.uint8)
        less[65:95, 23:27] = 255
        ok = np.zeros((100, 50), dtype=np.uint8)
        ok[15:95, 20:30] = 255

        self.assertEqual(classify_features(extract_features(missing, 180), rules)[0], "missing")
        self.assertEqual(classify_features(extract_features(less, 180), rules)[0], "less")
        self.assertEqual(classify_features(extract_features(ok, 180), rules)[0], "ok")


if __name__ == "__main__":
    unittest.main()
