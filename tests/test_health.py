import threading

from fastapi.testclient import TestClient
from PIL import Image
from test_catalog import ColorEncoder
from test_jobs import wait_for

from catalog_search.api import create_app
from catalog_search.catalog import index_catalog
from catalog_search.config import Settings


def test_http_available_during_model_loading_then_ready(tmp_path, monkeypatch):
    import catalog_search.encoder as module

    root = tmp_path / "photos"
    root.mkdir()
    Image.new("RGB", (20, 40), "red").save(root / "red.png")
    settings = Settings(catalog=root, storage=tmp_path / "var", index_settle_seconds=0)
    entered, release = threading.Event(), threading.Event()

    def load(settings):
        entered.set()
        assert release.wait(5)
        return ColorEncoder()

    monkeypatch.setattr(module, "DinoEncoder", load)
    with TestClient(create_app(settings)) as client:
        try:
            assert entered.wait(5)
            assert client.get("/").status_code == 200
            assert client.get("/health/live").status_code == 200
            response = client.get("/health/ready")
            assert response.status_code == 503
            assert response.json()["phase"] == "loading_model"
            response = client.post("/api/search", files={"file": ("q.png", b"query")})
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "3"
        finally:
            release.set()
        wait_for(lambda: client.get("/health/ready").status_code == 200)


def test_failed_model_is_live_but_not_ready(tmp_path, monkeypatch):
    import catalog_search.encoder as module

    def fail(settings):
        raise RuntimeError("missing model")

    monkeypatch.setattr(module, "DinoEncoder", fail)
    with TestClient(create_app(Settings(storage=tmp_path, index_interval=0))) as client:
        wait_for(lambda: client.get("/api/status").json()["phase"] == "model_error")
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503


def test_readiness_handles_empty_incompatible_and_corrupt_index(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    settings = Settings(catalog=root, storage=tmp_path / "var", index_interval=0)
    with TestClient(create_app(settings, encoder=ColorEncoder())) as client:
        assert client.get("/health/ready").json()["phase"] == "empty_catalog"
        Image.new("RGB", (20, 40), "red").save(root / "red.png")
        other = ColorEncoder()
        other.signature = "incompatible"
        index_catalog(settings, other)
        assert client.get("/health/ready").json()["phase"] == "incompatible_index"
        index_catalog(settings, ColorEncoder(), rebuild=True)
        assert client.get("/health/ready").status_code == 200
        settings.database.write_bytes(b"corrupt database")
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").json()["phase"] == "index_error"
