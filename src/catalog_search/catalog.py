"""Atomic catalog snapshots in SQLite and exact cosine search through NumPy."""

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np
from filelock import FileLock
from PIL import Image

from .config import MAX_IMAGE_BYTES, Settings
from .images import SUPPORTED_SUFFIXES, ImageError, decode_image, thumbnail_bytes
from .storage import SCHEMA_VERSION, CatalogError, connect, metadata


class Encoder(Protocol):
    dimension: int
    signature: str

    def encode(self, images: list[Image.Image]) -> np.ndarray: ...


class IndexCancelled(Exception):
    """Cooperative cancellation; the current transaction must be rolled back."""


@dataclass
class IndexReport:
    indexed: int = 0
    resumed: int = 0
    unchanged: int = 0
    removed: int = 0
    total: int = 0
    deferred: int = 0
    fast_skipped: int = 0
    hashed: int = 0
    full_verification: bool = False
    errors: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0


def catalog_status(settings: Settings) -> dict:
    if not settings.database.is_file():
        return {"ready": False, "count": 0, "generation": None}
    connection = connect(settings.database, readonly=True)
    try:
        connection.execute("BEGIN")
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not {"metadata", "images"} <= tables:
            return {"ready": False, "count": 0, "generation": None}
        info = metadata(connection)
        count = connection.execute("SELECT count(*) FROM images").fetchone()[0]
        return {
            "ready": bool(info.get("signature")) and count > 0,
            "count": count,
            "generation": info.get("generation"),
            "signature": info.get("signature"),
            "updated_at": info.get("updated_at"),
        }
    finally:
        connection.close()


def index_catalog(
    settings: Settings,
    encoder: Encoder,
    *,
    batch_size: int = 4,
    rebuild: bool = False,
    stop_event: threading.Event | None = None,
    settle_seconds: float = 0,
    verify: bool = False,
) -> IndexReport:
    if batch_size < 1:
        raise CatalogError("Размер пакета должен быть положительным")
    root = settings.catalog.resolve()
    if not root.is_dir():
        raise CatalogError(f"Папка каталога не найдена: {root}")
    settings.storage.mkdir(parents=True, exist_ok=True)
    with FileLock(str(settings.storage / "index.lock"), timeout=0):
        return _index(
            settings, root, encoder, batch_size, rebuild, stop_event, settle_seconds, verify
        )


def fingerprint(stat):
    return json.dumps([stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev])


def _index(settings, root, encoder, batch_size, rebuild, stop_event, settle_seconds, verify):
    started = time.perf_counter()
    report = IndexReport()
    connection = connect(settings.database)

    def check_cancelled():
        if stop_event is not None and stop_event.is_set():
            raise IndexCancelled

    try:
        check_cancelled()
        previous = metadata(connection)
        scan_started = time.time()
        report.full_verification = (
            verify
            or rebuild
            or not previous.get("last_full_scan")
            or scan_started - float(previous["last_full_scan"]) >= settings.index_verify_interval
        )
        if previous and not rebuild:
            if previous.get("signature") != encoder.signature:
                raise CatalogError(
                    "Изменилась модель или подготовка изображений. Нужен index --rebuild"
                )
            if previous.get("root") != str(root):
                raise CatalogError(
                    "Индекс относится к другой папке. Укажите другую --storage или --rebuild"
                )
        old = {
            row["path"]: row["content_hash"]
            for row in connection.execute("SELECT path, content_hash FROM images")
        }
        states = dict(connection.execute("SELECT path, fingerprint FROM file_state"))
        context = {
            "signature": encoder.signature,
            "dimension": str(encoder.dimension),
            "root": str(root),
            "rebuild": str(rebuild),
            "base_generation": previous.get("generation", ""),
        }
        if dict(connection.execute("SELECT key, value FROM staging_context")) != context:
            connection.execute("DELETE FROM staged_images")
            connection.execute("DELETE FROM staging_context")
            connection.executemany("INSERT INTO staging_context VALUES (?, ?)", context.items())
            connection.commit()
        staged = dict(connection.execute("SELECT path, content_hash FROM staged_images"))
        pending = []
        pending_images = []
        retained = set()
        selected = set()
        observed = {}

        def flush():
            check_cancelled()
            if not pending:
                return
            vectors = np.asarray(encoder.encode(pending_images), dtype=np.float32)
            if vectors.shape != (len(pending), encoder.dimension) or not np.isfinite(vectors).all():
                raise CatalogError("Некорректные эмбеддинги; предыдущий индекс сохранен")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if (norms < 1e-8).any():
                raise CatalogError("Модель вернула нулевой вектор")
            vectors = vectors / norms
            for item, vector in zip(pending, vectors, strict=True):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO staged_images VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (*item[:5], vector.astype("<f4").tobytes(), item[5], item[6]),
                )
                report.indexed += 1
            # Completed batches survive cancellation or failure of the next batch.
            connection.commit()
            pending.clear()
            pending_images.clear()

        for path in sorted(root.rglob("*")):
            check_cancelled()
            if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
                continue
            # Do not follow links to files or directories outside the configured catalog.
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            relative = path.relative_to(root).as_posix()
            try:
                before = path.stat()
                if settle_seconds and time.time() - before.st_mtime < settle_seconds:
                    retained.add(relative)
                    report.deferred += 1
                    continue
                stamp = fingerprint(before)
                if (
                    not report.full_verification
                    and relative in old
                    and states.get(relative) == stamp
                ):
                    retained.add(relative)
                    report.unchanged += 1
                    report.fast_skipped += 1
                    observed[relative] = stamp
                    continue
                with path.open("rb") as stream:
                    data = stream.read(MAX_IMAGE_BYTES + 1)
                after = path.stat()
                if stamp != fingerprint(after):
                    retained.add(relative)
                    report.deferred += 1
                    continue
                if len(data) > MAX_IMAGE_BYTES:
                    raise ImageError("Файл больше 20 МБ")
                digest = hashlib.sha256(data).hexdigest()
                report.hashed += 1
                observed[relative] = stamp
                if not rebuild and old.get(relative) == digest:
                    retained.add(relative)
                    report.unchanged += 1
                    continue
                if staged.get(relative) == digest:
                    selected.add(relative)
                    retained.add(relative)
                    report.resumed += 1
                    continue
                image = decode_image(data)
                identifier = hashlib.sha256(relative.encode("utf-8")).hexdigest()
                pending.append(
                    (
                        identifier,
                        relative,
                        digest,
                        image.width,
                        image.height,
                        thumbnail_bytes(image),
                        stamp,
                    )
                )
                pending_images.append(image)
                retained.add(relative)
                selected.add(relative)
            except (OSError, ImageError) as exc:
                report.errors.append({"path": relative, "error": str(exc)})
            if len(pending) >= batch_size:
                flush()
        flush()
        # A file can change during inference, long after its initial read.
        for relative, stamp in list(observed.items()):
            check_cancelled()
            path = root / relative
            try:
                stable = (
                    not path.is_symlink()
                    and path.resolve().is_relative_to(root)
                    and fingerprint(path.stat()) == stamp
                )
            except OSError:
                stable = False
            if not stable:
                selected.discard(relative)
                observed.pop(relative)
                retained.add(relative)
                report.deferred += 1
        if rebuild and report.deferred:
            raise CatalogError(
                "Файлы меняются во время rebuild; повторите проход. Пакеты сохранены"
            )
        check_cancelled()
        connection.execute("BEGIN IMMEDIATE")
        if rebuild:
            connection.execute("DELETE FROM images")
            connection.execute("DELETE FROM file_state")
        connection.executemany(
            """INSERT OR REPLACE INTO images
            SELECT id, path, content_hash, width, height, embedding, thumbnail
            FROM staged_images WHERE path=?""",
            [(p,) for p in selected],
        )
        connection.executemany("INSERT OR REPLACE INTO file_state VALUES (?, ?)", observed.items())
        removed = set(old) - retained
        connection.executemany("DELETE FROM images WHERE path = ?", [(p,) for p in removed])
        connection.execute("DELETE FROM file_state WHERE path NOT IN (SELECT path FROM images)")
        report.removed = len(removed)
        report.total = connection.execute("SELECT count(*) FROM images").fetchone()[0]
        changed = rebuild or selected or report.removed or not previous
        if changed:
            info = {
                "signature": encoder.signature,
                "dimension": str(encoder.dimension),
                "root": str(root),
                "generation": uuid.uuid4().hex,
                "updated_at": str(time.time()),
                "schema_version": str(SCHEMA_VERSION),
            }
            connection.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", info.items())
        if report.full_verification and not report.deferred and not report.errors:
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('last_full_scan', ?)", (str(scan_started),)
            )
        check_cancelled()
        connection.execute("DELETE FROM staged_images")
        connection.execute("DELETE FROM staging_context")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    report.elapsed_seconds = round(time.perf_counter() - started, 3)
    return report


class SearchIndex:
    def __init__(self, settings: Settings, signature: str):
        connection = connect(settings.database, readonly=True)
        try:
            connection.execute("BEGIN")
            info = metadata(connection)
            if info.get("signature") != signature:
                raise CatalogError("Индекс несовместим с моделью. Выполните index --rebuild")
            self.generation = info["generation"]
            self.dimension = int(info["dimension"])
            records = connection.execute(
                "SELECT id, path, content_hash, width, height, embedding FROM images ORDER BY path"
            ).fetchall()
        finally:
            connection.close()
        self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        self.items = []
        if records:
            vectors = np.stack([np.frombuffer(row["embedding"], dtype="<f4") for row in records])
            if vectors.shape != (len(records), self.dimension) or not np.isfinite(vectors).all():
                raise CatalogError("Поврежден индекс. Выполните index --rebuild")
            self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            self.items = [
                {key: row[key] for key in row.keys() if key != "embedding"} for row in records
            ]

    def search(
        self, vector: np.ndarray, *, top_k: int = 10, exclude_hash: str | None = None
    ) -> list[dict]:
        if not 1 <= top_k <= 100:
            raise CatalogError("top_k должен быть от 1 до 100")
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dimension or not np.isfinite(vector).all():
            raise CatalogError("Некорректный вектор запроса")
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            raise CatalogError("Нулевой вектор запроса")
        if not self.items:
            return []
        # Fetch extra candidates when all byte-identical copies should be excluded.
        duplicates = sum(item["content_hash"] == exclude_hash for item in self.items)
        count = min(len(self.items), top_k + duplicates)
        scores = self.vectors @ (vector[0] / norm)
        positions = np.argpartition(-scores, count - 1)[:count]
        positions = positions[np.lexsort((positions, -scores[positions]))]
        results = []
        for position in positions:
            score = scores[position]
            item = self.items[int(position)]
            if item["content_hash"] == exclude_hash:
                continue
            results.append(
                {
                    "id": item["id"],
                    "path": item["path"],
                    "score": round(float(np.clip(score, -1, 1)), 6),
                    "thumbnail_url": f"/api/images/{item['id']}/thumbnail",
                }
            )
            if len(results) == top_k:
                break
        return results


def report_json(report: IndexReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, indent=2)
