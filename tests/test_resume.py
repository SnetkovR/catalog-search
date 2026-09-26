import threading

import pytest
from PIL import Image
from test_catalog import ColorEncoder, catalog  # noqa: F401

from catalog_search.catalog import IndexCancelled, catalog_status, index_catalog
from catalog_search.storage import connect


class FailSecondBatch(ColorEncoder):
    def encode(self, images):
        if self.encoded:
            raise RuntimeError("simulated crash")
        return super().encode(images)


def test_completed_batches_survive_failed_rebuild_and_resume(catalog):  # noqa: F811
    settings, encoder = catalog
    index_catalog(settings, encoder)
    before = catalog_status(settings)
    with pytest.raises(RuntimeError):
        index_catalog(settings, FailSecondBatch(), rebuild=True, batch_size=1)
    assert catalog_status(settings) == before
    fresh_process_encoder = ColorEncoder()
    report = index_catalog(settings, fresh_process_encoder, rebuild=True, batch_size=1)
    assert (report.resumed, report.indexed, report.total) == (1, 1, 2)
    assert fresh_process_encoder.encoded == 1
    with connect(settings.database) as db:
        assert db.execute("SELECT count(*) FROM staged_images").fetchone()[0] == 0


def test_resume_revalidates_changed_deleted_files_and_model(catalog):  # noqa: F811
    settings, _ = catalog
    with pytest.raises(RuntimeError):
        index_catalog(settings, FailSecondBatch(), batch_size=1)
    Image.new("RGB", (20, 40), "green").save(settings.catalog / "blue.png")
    report = index_catalog(settings, ColorEncoder(), batch_size=1)
    assert (report.resumed, report.indexed) == (0, 2)
    with pytest.raises(RuntimeError):
        index_catalog(settings, FailSecondBatch(), rebuild=True, batch_size=1)
    (settings.catalog / "blue.png").unlink()
    encoder = ColorEncoder()
    encoder.signature = "new-model"
    report = index_catalog(settings, encoder, rebuild=True)
    assert (report.resumed, report.indexed, report.total) == (0, 1, 1)


def test_cancelled_batch_is_saved_without_publishing(catalog):  # noqa: F811
    settings, _ = catalog
    stop = threading.Event()

    class CancelAfterEncode(ColorEncoder):
        def encode(self, images):
            result = super().encode(images)
            stop.set()
            return result

    with pytest.raises(IndexCancelled):
        index_catalog(settings, CancelAfterEncode(), batch_size=1, stop_event=stop)
    assert catalog_status(settings)["count"] == 0
    report = index_catalog(settings, ColorEncoder(), batch_size=1)
    assert (report.resumed, report.indexed) == (1, 1)


def test_file_changed_during_inference_is_not_published(catalog):  # noqa: F811
    settings, encoder = catalog
    index_catalog(settings, encoder)
    before = catalog_status(settings)
    Image.new("RGB", (20, 40), "green").save(settings.catalog / "red.png")

    class EditingEncoder(ColorEncoder):
        def encode(self, images):
            result = super().encode(images)
            Image.new("RGB", (20, 40), "blue").save(settings.catalog / "red.png")
            return result

    report = index_catalog(settings, EditingEncoder())
    assert report.deferred == 1
    assert catalog_status(settings) == before
    assert index_catalog(settings, ColorEncoder()).indexed == 1
