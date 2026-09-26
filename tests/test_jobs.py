import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from filelock import FileLock
from PIL import Image
from test_catalog import ColorEncoder

from catalog_search.catalog import catalog_status, index_catalog
from catalog_search.config import Settings
from catalog_search.jobs import IndexingJob, PriorityEncoder


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for background work")


@pytest.fixture
def settings(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    return Settings(
        catalog=root,
        storage=tmp_path / "var",
        threads=1,
        index_interval=0.05,
        index_settle_seconds=0,
    )


def test_job_discovers_additions_changes_and_deletions(settings):
    encoder = ColorEncoder()
    job = IndexingJob(settings, PriorityEncoder(encoder))
    job.start()
    try:
        wait_for(lambda: job.snapshot()["report"] is not None)
        assert catalog_status(settings)["count"] == 0
        Image.new("RGB", (20, 40), "red").save(settings.catalog / "new.png")
        wait_for(lambda: catalog_status(settings)["count"] == 1)
        generation = catalog_status(settings)["generation"]
        Image.new("RGB", (20, 40), "blue").save(settings.catalog / "new.png")
        wait_for(lambda: catalog_status(settings)["generation"] != generation)
        assert encoder.encoded == 2
        (settings.catalog / "new.png").unlink()
        wait_for(lambda: catalog_status(settings)["count"] == 0)
    finally:
        job.stop()
    assert job.snapshot()["state"] == "stopped"


def test_job_recovers_when_catalog_appears(settings):
    settings.catalog.rmdir()
    job = IndexingJob(settings, PriorityEncoder(ColorEncoder()))
    job.start()
    try:
        wait_for(lambda: job.snapshot()["state"] == "error")
        assert "не найдена" in job.snapshot()["error"]
        settings.catalog.mkdir()
        Image.new("RGB", (20, 40), "red").save(settings.catalog / "new.png")
        wait_for(lambda: catalog_status(settings)["count"] == 1)
        wait_for(lambda: job.snapshot()["error"] is None)
    finally:
        job.stop()


def test_other_indexer_lock_is_retried(settings):
    settings.storage.mkdir()
    job = IndexingJob(settings, PriorityEncoder(ColorEncoder()))
    try:
        with FileLock(str(settings.storage / "index.lock")):
            job.start()
            wait_for(lambda: job.snapshot()["error"] is not None)
            assert job.snapshot()["state"] == "waiting"
        wait_for(lambda: job.snapshot()["report"] is not None)
    finally:
        job.stop()


def test_unsettled_file_preserves_old_entry(settings):
    path = settings.catalog / "red.png"
    Image.new("RGB", (20, 40), "red").save(path)
    encoder = ColorEncoder()
    index_catalog(settings, encoder)
    generation = catalog_status(settings)["generation"]
    path.write_bytes(b"incomplete upload")
    report = index_catalog(settings, encoder, settle_seconds=10)
    assert report.deferred == 1
    assert report.removed == 0
    assert catalog_status(settings)["generation"] == generation
    Image.new("RGB", (20, 40), "blue").save(path)
    os.utime(path, (time.time() - 20, time.time() - 20))
    assert index_catalog(settings, encoder, settle_seconds=10).indexed == 1


def test_shutdown_cancels_transaction(settings):
    Image.new("RGB", (20, 40), "red").save(settings.catalog / "old.png")
    index_catalog(settings, ColorEncoder())
    generation = catalog_status(settings)["generation"]
    Image.new("RGB", (20, 40), "blue").save(settings.catalog / "new.png")
    entered, release = threading.Event(), threading.Event()

    class BlockingEncoder(ColorEncoder):
        def encode(self, images):
            entered.set()
            assert release.wait(5)
            return super().encode(images)

    job = IndexingJob(settings, PriorityEncoder(BlockingEncoder()))
    job.start()
    try:
        assert entered.wait(5)
        assert catalog_status(settings)["count"] == 1  # Uncommitted rows are invisible.
        with ThreadPoolExecutor() as executor:
            stopped = executor.submit(job.stop)
            assert job._stop.wait(2)
            release.set()
            stopped.result(timeout=5)
        assert catalog_status(settings)["generation"] == generation
        assert catalog_status(settings)["count"] == 1
    finally:
        release.set()
        job.stop()


def test_queries_take_priority_between_background_batches():
    entered, release = threading.Event(), threading.Event()
    calls = []

    class RecordingEncoder(ColorEncoder):
        def encode(self, images):
            calls.append(images[0])
            if images == ["first-background"]:
                entered.set()
                assert release.wait(5)
            return images

    shared = PriorityEncoder(RecordingEncoder())
    background = shared.background(threading.Event())
    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(background.encode, ["first-background"])
        try:
            assert entered.wait(5)
            query = executor.submit(shared.encode, ["query"])
            wait_for(lambda: shared._foreground_waiters == 1)
            second = executor.submit(background.encode, ["second-background"])
        finally:
            release.set()
        for future in (first, query, second):
            future.result(timeout=5)
    assert calls == ["first-background", "query", "second-background"]


def test_disabled_job_does_not_create_database(tmp_path):
    settings = Settings(storage=tmp_path, index_interval=0)
    job = IndexingJob(settings, PriorityEncoder(ColorEncoder()))
    job.start()
    job.stop()
    assert job.snapshot()["state"] == "disabled"
    assert not settings.database.exists()
