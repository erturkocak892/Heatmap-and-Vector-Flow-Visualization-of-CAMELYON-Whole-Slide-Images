import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from server.app import app, _get_models, _hovernet_assets_status
from server.inference_service import InferenceJobManager


class AppRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_serves_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("CAMELYON", response.text)

    def test_app_page_serves_html(self):
        response = self.client.get("/app")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))

    def test_slides_endpoint_returns_list(self):
        response = self.client.get("/api/slides")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_models_include_availability_metadata(self):
        models = _get_models()
        hovernet_ready, _ = _hovernet_assets_status()

        self.assertTrue(models, "Expected at least one model configuration")
        self.assertTrue(all("available" in model for model in models))

        hovernet = next(model for model in models if model["id"] == "hovernet")
        self.assertEqual(hovernet["available"], hovernet_ready)

        mock_models = [model for model in models if model["id"].startswith("mock-")]
        self.assertTrue(all(model["available"] for model in mock_models))


class InferenceManagerTests(unittest.TestCase):
    def test_start_inference_requires_hovernet_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = InferenceJobManager(Path(tmpdir))
            fake_slide_entry = SimpleNamespace(slide_id="slide-1")

            with self.assertRaises(FileNotFoundError):
                manager.start_inference(fake_slide_entry, model_id="hovernet")

    def test_extract_region_writes_png_for_roi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = InferenceJobManager(Path(tmpdir))

            class FakeSlide:
                dimensions = (2048, 1024)
                level_count = 3
                level_downsamples = (1, 2, 4)
                level_dimensions = ((2048, 1024), (1024, 512), (512, 256))

                def read_region(self, location, level, size):
                    return Image.new("RGBA", size, (255, 0, 0, 255))

            fake_entry = SimpleNamespace(slide=FakeSlide())
            output_path = Path(tmpdir) / "region.png"

            manager._extract_region(
                fake_entry,
                {"x": 100, "y": 50, "width": 300, "height": 200},
                output_path,
            )

            self.assertTrue(output_path.exists())
            with Image.open(output_path) as saved:
                self.assertEqual(saved.size, (300, 200))
                self.assertEqual(saved.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
