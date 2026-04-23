import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from weld_cv.labels import expected_label


class LabelTests(unittest.TestCase):
    def test_reads_label_from_folder_and_filename(self):
        self.assertEqual(expected_label(Path("data/raw/ok/Image_1.bmp")), "ok")
        self.assertEqual(expected_label(Path("data/raw/ng/less (1).bmp")), "less")
        self.assertEqual(expected_label(Path("data/raw/ng/missing (1).bmp")), "missing")
        self.assertEqual(expected_label(Path("data/raw/unknown/ok (1).bmp")), "ok")
        self.assertEqual(expected_label(Path("data/raw/unknown/Image_1.bmp")), "unknown")


if __name__ == "__main__":
    unittest.main()

