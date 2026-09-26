import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_catalog import ColorEncoder

from catalog_search.api import create_app
from catalog_search.catalog import index_catalog
from catalog_search.config import Settings


@pytest.fixture
def web(tmp_path):
    root = tmp_path / "catalog"
    root.mkdir()
    Image.new("RGB", (30, 50), "red").save(root / "red.png")
    Image.new("RGB", (30, 50), "blue").save(root / "blue.png")
    settings = Settings(catalog=root, storage=tmp_path / "var", threads=1, index_interval=0)
    encoder = ColorEncoder()
    index_catalog(settings, encoder)
    app = create_app(settings, encoder=encoder)
    with TestClient(app) as client:
        yield client, settings, encoder, app


def test_upload_search_and_catalog_neighbors(web):
    client, settings, _, _ = web
    data = (settings.catalog / "red.png").read_bytes()
    response = client.post("/api/search", files={"file": ("red.png", data)}, data={"top_k": 1})
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["path"] == "red.png"
    assert client.get(result["thumbnail_url"]).headers["content-type"] == "image/jpeg"
    response = client.post(f"/api/search/catalog/{result['id']}")
    assert [r["path"] for r in response.json()["results"]] == ["blue.png"]
    assert client.get("/api/status").json()["device"] == "cpu"


def test_invalid_upload_crop_and_limits(web):
    client, settings, _, _ = web
    data = (settings.catalog / "red.png").read_bytes()
    assert client.post("/api/search", files={"file": ("bad.png", b"bad")}).status_code == 422
    for crop in ["[1,0,0,1]", "{}", "[0,0,NaN,1]", "not json"]:
        assert (
            client.post(
                "/api/search", files={"file": ("red.png", data)}, data={"crop": crop}
            ).status_code
            == 422
        )
    assert (
        client.post("/api/search", files={"file": ("red.png", data)}, data={"top_k": 0}).status_code
        == 422
    )
    assert client.get("/api/images/not-a-file/thumbnail").status_code == 404


def test_crop_changes_retrieval_target(web):
    client, _, _, _ = web
    image = Image.new("RGB", (30, 60), "red")
    image.paste("blue", (0, 30, 30, 60))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    response = client.post(
        "/api/search",
        files={"file": ("query.png", buffer.getvalue())},
        data={"crop": "[0,0.5,1,1]", "top_k": 1},
    )
    assert response.json()["results"][0]["path"] == "blue.png"


def test_index_refresh_after_incremental_update(web):
    client, settings, encoder, _ = web
    data = (settings.catalog / "red.png").read_bytes()
    client.post("/api/search", files={"file": ("query.png", data)})
    (settings.catalog / "red.png").unlink()
    index_catalog(settings, encoder)
    response = client.post("/api/search", files={"file": ("query.png", data)})
    assert response.json()["count"] == 1
    assert response.json()["results"][0]["path"] == "blue.png"


def test_unchanged_scan_reuses_loaded_search_index(web):
    _, settings, encoder, app = web
    initial = app.state.service.refresh()
    index_catalog(settings, encoder)
    assert app.state.service.refresh() is initial


def test_busy_inference_returns_retryable_error(web):
    client, settings, _, app = web
    data = (settings.catalog / "red.png").read_bytes()
    with app.state.service.gate:
        assert client.post("/api/search", files={"file": ("red.png", data)}).status_code == 503


def test_empty_catalog(tmp_path):
    app = create_app(Settings(storage=tmp_path, index_interval=0), encoder=ColorEncoder())
    with TestClient(app) as client:
        assert client.get("/api/status").json()["ready"] is False
        assert client.get("/api/catalog").json()["items"] == []
        assert client.post("/api/search", files={"file": ("a.png", b"x")}).status_code == 409


def test_oversized_body_and_chunked_upload_are_rejected(web, monkeypatch):
    import catalog_search.api as api

    monkeypatch.setattr(api, "MAX_IMAGE_BYTES", 1024)
    client, _, _, _ = web
    assert client.post("/api/search", files={"file": ("big.jpg", b"x" * 2048)}).status_code == 413
    chunks = iter(
        [
            b'--boundary\r\nContent-Disposition: form-data; name="file"; '
            b'filename="big.jpg"\r\n\r\n',
            b"x" * 70_000,
            b"\r\n--boundary--\r\n",
        ]
    )
    response = client.post(
        "/api/search",
        content=chunks,
        headers={"Content-Type": "multipart/form-data; boundary=boundary"},
    )
    assert response.status_code == 413
