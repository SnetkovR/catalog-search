import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
from PIL import Image
from test_catalog import ColorEncoder
from test_jobs import wait_for

from catalog_search.api import create_app
from catalog_search.catalog import index_catalog
from catalog_search.config import Settings


def test_server_indexes_new_files_and_releases_worker_on_shutdown(tmp_path):
    root = tmp_path / "catalog"
    root.mkdir()
    settings = Settings(
        catalog=root,
        storage=tmp_path / "var",
        index_interval=0.05,
        index_settle_seconds=0,
        threads=1,
    )
    app = create_app(settings, encoder=ColorEncoder())
    with TestClient(app) as client:
        wait_for(lambda: client.get("/api/status").json()["indexing"]["report"] is not None)
        assert client.get("/api/status").json()["ready"] is False
        Image.new("RGB", (20, 40), "red").save(root / "new.png")
        wait_for(lambda: client.get("/api/status").json()["ready"])
        assert client.get("/api/catalog").json()["total"] == 1
        response = client.post(
            "/api/search", files={"file": ("query.png", (root / "new.png").read_bytes())}
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["path"] == "new.png"
        (root / "new.png").unlink()
        wait_for(lambda: client.get("/api/status").json()["count"] == 0)
        assert client.get("/api/catalog").json()["items"] == []
    assert app.state.indexing_job.snapshot()["state"] == "stopped"


def test_existing_catalog_stays_available_while_background_model_is_busy(tmp_path):
    root = tmp_path / "catalog"
    root.mkdir()
    settings = Settings(
        catalog=root, storage=tmp_path / "var", index_interval=60, index_settle_seconds=0, threads=1
    )
    Image.new("RGB", (20, 40), "red").save(root / "old.png")
    index_catalog(settings, ColorEncoder())
    Image.new("RGB", (20, 40), "blue").save(root / "new.png")
    entered, release = threading.Event(), threading.Event()

    class SlowEncoder(ColorEncoder):
        def encode(self, images):
            if threading.current_thread().name == "catalog-indexer":
                entered.set()
                assert release.wait(5)
            return super().encode(images)

    app = create_app(settings, encoder=SlowEncoder())
    with TestClient(app) as client:
        try:
            assert entered.wait(5)
            status = client.get("/api/status").json()
            assert status["ready"] is True
            assert status["count"] == 1
            assert status["indexing"]["state"] == "running"
            item = client.get("/api/catalog").json()["items"][0]
            assert client.get(item["thumbnail_url"]).status_code == 200
            assert client.post(f"/api/search/catalog/{item['id']}").status_code == 200
            with ThreadPoolExecutor() as executor:
                response = executor.submit(
                    client.post,
                    "/api/search",
                    files={"file": ("query.png", (root / "old.png").read_bytes())},
                )
                try:
                    wait_for(lambda: app.state.service.encoder._foreground_waiters == 1)
                finally:
                    release.set()
                assert response.result(timeout=5).status_code == 200
            wait_for(lambda: client.get("/api/status").json()["count"] == 2)
        finally:
            release.set()
