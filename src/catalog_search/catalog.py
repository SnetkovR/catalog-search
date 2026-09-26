"""Atomic catalog snapshots in SQLite; exact cosine search through CPU FAISS."""

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
from filelock import FileLock
from PIL import Image

from .config import MAX_IMAGE_BYTES, Settings
from .images import SUPPORTED_SUFFIXES, ImageError, decode_image, thumbnail_bytes


class Encoder(Protocol):
    dimension: int
    signature: str

    def encode(self, images: list[Image.Image]) -> np.ndarray: ...


class CatalogError(ValueError):
    pass


@dataclass
class IndexReport:
    indexed: int = 0
    unchanged: int = 0
    removed: int = 0
    total: int = 0
    errors: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0


def connect(database: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        if not database.is_file():
            raise CatalogError("Каталог еще не проиндексирован. Выполните catalog-search index")
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    else:
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
    connection.row_factory = sqlite3.Row
    return connection


def metadata(connection: sqlite3.Connection) -> dict:
    return dict(connection.execute("SELECT key, value FROM metadata").fetchall())


def catalog_status(settings: Settings) -> dict:
    if not settings.database.is_file():
        return {"ready": False, "count": 0, "generation": None}
    connection = connect(settings.database, readonly=True)
    try:
        connection.execute("BEGIN")
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
    settings: Settings, encoder: Encoder, *, batch_size: int = 4, rebuild: bool = False
) -> IndexReport:
    if batch_size < 1:
        raise CatalogError("Размер пакета должен быть положительным")
    root = settings.catalog.resolve()
    if not root.is_dir():
        raise CatalogError(f"Папка каталога не найдена: {root}")
    settings.storage.mkdir(parents=True, exist_ok=True)
    with FileLock(str(settings.storage / "index.lock"), timeout=0):
        return _index(settings, root, encoder, batch_size, rebuild)


def _index(settings, root, encoder, batch_size, rebuild):
    started = time.perf_counter()
    report = IndexReport()
    connection = connect(settings.database)
    try:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, content_hash TEXT NOT NULL,
                width INTEGER NOT NULL, height INTEGER NOT NULL,
                embedding BLOB NOT NULL, thumbnail BLOB NOT NULL
            );
        """)
        connection.execute("BEGIN IMMEDIATE")
        previous = metadata(connection)
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
        if rebuild:
            connection.execute("DELETE FROM images")
        pending = []
        pending_images = []
        retained = set()

        def flush():
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
                    INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET path=excluded.path,
                        content_hash=excluded.content_hash, width=excluded.width,
                        height=excluded.height, embedding=excluded.embedding,
                        thumbnail=excluded.thumbnail
                """,
                    (*item[:5], vector.astype("<f4").tobytes(), item[5]),
                )
                report.indexed += 1
            pending.clear()
            pending_images.clear()

        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
                continue
            # Do not follow links to files or directories outside the configured catalog.
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            relative = path.relative_to(root).as_posix()
            try:
                with path.open("rb") as stream:
                    data = stream.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    raise ImageError("Файл больше 20 МБ")
                digest = hashlib.sha256(data).hexdigest()
                if not rebuild and old.get(relative) == digest:
                    retained.add(relative)
                    report.unchanged += 1
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
                    )
                )
                pending_images.append(image)
                retained.add(relative)
            except (OSError, ImageError) as exc:
                report.errors.append({"path": relative, "error": str(exc)})
            if len(pending) >= batch_size:
                flush()
        flush()
        removed = set(old) - retained
        connection.executemany("DELETE FROM images WHERE path = ?", [(p,) for p in removed])
        report.removed = len(removed)
        report.total = connection.execute("SELECT count(*) FROM images").fetchone()[0]
        changed = rebuild or report.indexed or report.removed or not previous
        if changed:
            info = {
                "signature": encoder.signature,
                "dimension": str(encoder.dimension),
                "root": str(root),
                "generation": uuid.uuid4().hex,
                "updated_at": str(time.time()),
                "schema_version": "1",
            }
            connection.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", info.items())
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
        faiss.omp_set_num_threads(settings.threads)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.items = []
        if records:
            vectors = np.stack([np.frombuffer(row["embedding"], dtype="<f4") for row in records])
            if vectors.shape != (len(records), self.dimension) or not np.isfinite(vectors).all():
                raise CatalogError("Поврежден индекс. Выполните index --rebuild")
            self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
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
        scores, positions = self.index.search(np.ascontiguousarray(vector / norm), count)
        results = []
        for score, position in zip(scores[0], positions[0], strict=True):
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
