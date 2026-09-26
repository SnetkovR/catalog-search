"""Periodic indexing with one shared CPU model and foreground query priority."""

import logging
import threading
import time
from dataclasses import asdict

from filelock import Timeout

from .catalog import IndexCancelled, index_catalog

logger = logging.getLogger(__name__)


class PriorityEncoder:
    """Serialize model access, yielding to queued queries between background batches."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.signature = encoder.signature
        self.dimension = encoder.dimension
        self._condition = threading.Condition()
        self._active = False
        self._foreground_waiters = 0

    def encode(self, images):
        with self._condition:
            self._foreground_waiters += 1
            try:
                self._condition.wait_for(lambda: not self._active)
                self._active = True
            finally:
                self._foreground_waiters -= 1
        try:
            return self.encoder.encode(images)
        finally:
            self._release()

    def _release(self):
        with self._condition:
            self._active = False
            self._condition.notify_all()

    def background(self, stop_event):
        return BackgroundEncoder(self, stop_event)


class BackgroundEncoder:
    def __init__(self, shared, stop_event):
        self.shared = shared
        self.stop_event = stop_event
        self.signature = shared.signature
        self.dimension = shared.dimension

    def encode(self, images):
        shared = self.shared
        with shared._condition:
            while shared._active or shared._foreground_waiters:
                if self.stop_event.is_set():
                    raise IndexCancelled
                shared._condition.wait(timeout=0.1)
            if self.stop_event.is_set():
                raise IndexCancelled
            shared._active = True
        try:
            return shared.encoder.encode(images)
        finally:
            shared._release()


class IndexingJob:
    """One polling thread, started and stopped with the web application's lifespan."""

    def __init__(self, settings, shared_encoder):
        self.settings = settings
        self._stop = threading.Event()
        self.encoder = shared_encoder.background(self._stop)
        self._state_lock = threading.Lock()
        self._thread = None
        self._state = {
            "enabled": settings.index_interval > 0,
            "state": "waiting" if settings.index_interval > 0 else "disabled",
            "interval_seconds": settings.index_interval,
            "last_started_at": None,
            "last_finished_at": None,
            "next_run_at": None,
            "report": None,
            "error": None,
        }

    def snapshot(self):
        with self._state_lock:
            return self._state.copy()

    def _update(self, **values):
        with self._state_lock:
            self._state.update(values)

    def start(self):
        if self.settings.index_interval <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="catalog-indexer", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            self._update(state="running", last_started_at=time.time(), next_run_at=None, error=None)
            try:
                report = index_catalog(
                    self.settings,
                    self.encoder,
                    batch_size=self.settings.index_batch_size,
                    stop_event=self._stop,
                    settle_seconds=self.settings.index_settle_seconds,
                )
                self._update(state="idle", report=asdict(report))
            except IndexCancelled:
                break
            except Timeout:
                self._update(state="waiting", error="Другой процесс обновляет каталог")
            except Exception as exc:
                logger.warning("Background indexing failed: %s", exc, exc_info=True)
                self._update(state="error", error=str(exc))
            self._update(
                last_finished_at=time.time(), next_run_at=time.time() + self.settings.index_interval
            )
            if self._stop.wait(self.settings.index_interval):
                break
        self._update(state="stopped", next_run_at=None)
