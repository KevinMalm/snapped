import unittest
import snapped


class TestLibrarySetup(unittest.TestCase):
    def test_label(self):
        self.assertEqual(snapped.__label__, "snapped")

    def test_version(self):
        self.assertTrue(snapped.__version__)


if __name__ == "__main__":
    unittest.main()
