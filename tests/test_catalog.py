import shutil

import numpy as np
import pytest
from PIL import Image

from catalog_search.catalog import CatalogError, SearchIndex, catalog_status, index_catalog
from catalog_search.config import Settings


class ColorEncoder:
    """Deterministic test double; production always uses DINOv2."""

    dimension = 3
    signature = "test-colors-v1"

    def __init__(self):
        self.encoded = 0

    def encode(self, images):
        self.encoded += len(images)
        return np.stack([np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) for image in images])


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    Image.new("RGB", (20, 40), "red").save(root / "red.png")
    Image.new("RGB", (20, 40), "blue").save(root / "blue.png")
    return Settings(catalog=root, storage=tmp_path / "index", threads=1), ColorEncoder()


def test_incremental_update_delete_and_restart(catalog):
    settings, encoder = catalog
    assert index_catalog(settings, encoder).indexed == 2
    generation = catalog_status(settings)["generation"]
    assert index_catalog(settings, encoder).unchanged == 2
    assert encoder.encoded == 2
    assert catalog_status(settings)["generation"] == generation
    (settings.catalog / "blue.png").unlink()
    Image.new("RGB", (20, 40), "green").save(settings.catalog / "red.png")
    report = index_catalog(settings, encoder)
    assert (report.indexed, report.removed, report.total) == (1, 1, 1)
    index = SearchIndex(settings, encoder.signature)
    assert index.search(np.array([0, 1, 0]), top_k=10)[0]["path"] == "red.png"


def test_search_excludes_all_identical_files(catalog):
    settings, encoder = catalog
    shutil.copy(settings.catalog / "red.png", settings.catalog / "red-copy.png")
    index_catalog(settings, encoder)
    index = SearchIndex(settings, encoder.signature)
    digest = next(item["content_hash"] for item in index.items if item["path"] == "red.png")
    assert index.search(np.array([1, 0, 0]), top_k=1)[0]["score"] == 1
    assert [item["path"] for item in index.search(np.array([1, 0, 0]), exclude_hash=digest)] == [
        "blue.png"
    ]


def test_failed_rebuild_preserves_previous_snapshot(catalog):
    settings, encoder = catalog
    index_catalog(settings, encoder)
    generation = catalog_status(settings)["generation"]

    class BrokenEncoder(ColorEncoder):
        def encode(self, images):
            raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError):
        index_catalog(settings, BrokenEncoder(), rebuild=True)
    assert catalog_status(settings)["generation"] == generation
    assert len(SearchIndex(settings, encoder.signature).items) == 2


def test_invalid_changed_file_is_removed_and_reported(catalog):
    settings, encoder = catalog
    index_catalog(settings, encoder)
    (settings.catalog / "red.png").write_bytes(b"corrupt")
    report = index_catalog(settings, encoder)
    assert (report.total, report.removed, len(report.errors)) == (1, 1, 1)


def test_signature_and_root_mismatch_require_explicit_rebuild(catalog, tmp_path):
    settings, encoder = catalog
    index_catalog(settings, encoder)
    encoder.signature = "other-version"
    with pytest.raises(CatalogError, match="rebuild"):
        index_catalog(settings, encoder)
    with pytest.raises(CatalogError, match="rebuild"):
        SearchIndex(settings, encoder.signature)
    other = tmp_path / "other"
    other.mkdir()
    encoder.signature = "test-colors-v1"
    with pytest.raises(CatalogError, match="другой папке"):
        index_catalog(Settings(catalog=other, storage=settings.storage), encoder)


def test_missing_catalog_does_not_erase_existing_index(catalog):
    settings, encoder = catalog
    index_catalog(settings, encoder)
    missing = Settings(catalog=settings.catalog / "missing", storage=settings.storage)
    with pytest.raises(CatalogError, match="не найдена"):
        index_catalog(missing, encoder)
    assert catalog_status(settings)["count"] == 2
