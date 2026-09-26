"""Versioned SQLite storage, consistent backups and offline recovery."""

import os
import sqlite3
import tempfile
import uuid
from contextlib import closing
from pathlib import Path

import numpy as np
from filelock import FileLock

SCHEMA_VERSION = 2


class CatalogError(ValueError):
    pass


def metadata(connection):
    return dict(connection.execute("SELECT key, value FROM metadata").fetchall())


def schema_version(connection):
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "metadata" in tables:
        declared = int(metadata(connection).get("schema_version", version or 1))
        if version and version != declared:
            raise CatalogError("Версии схемы SQLite и метаданных не совпадают")
        version = declared
    if version > SCHEMA_VERSION or version < 0:
        raise CatalogError(f"Неподдерживаемая версия схемы SQLite: {version}")
    return version


def snapshot(source, destination):
    """SQLite backup includes committed WAL pages, unlike copying the main file."""
    with closing(sqlite3.connect(destination)) as target:
        source.backup(target)


def migrate(connection, database):
    version = schema_version(connection)
    if version == SCHEMA_VERSION:
        return
    if version:
        backup = database.with_name(
            f"{database.stem}.before-v{SCHEMA_VERSION}-{uuid.uuid4().hex}.sqlite3"
        )
        snapshot(connection, backup)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute("""CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, content_hash TEXT NOT NULL,
            width INTEGER NOT NULL, height INTEGER NOT NULL,
            embedding BLOB NOT NULL, thumbnail BLOB NOT NULL)""")
        connection.execute("""CREATE TABLE file_state (
            path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL)""")
        connection.execute(
            "CREATE TABLE staging_context (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute("""CREATE TABLE staged_images (
            id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, content_hash TEXT NOT NULL,
            width INTEGER NOT NULL, height INTEGER NOT NULL,
            embedding BLOB NOT NULL, thumbnail BLOB NOT NULL, fingerprint TEXT NOT NULL)""")
        # An empty database has no published index yet.
        if metadata(connection):
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def connect(database: Path, *, readonly=False):
    connection = None
    try:
        if readonly:
            if not database.is_file():
                raise CatalogError("Каталог еще не проиндексирован. Выполните catalog-search index")
            connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            database.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(database, timeout=30)
        connection.row_factory = sqlite3.Row
        if readonly:
            schema_version(connection)
        else:
            migrate(connection, database)
            connection.execute("PRAGMA journal_mode=WAL")
        return connection
    except (sqlite3.DatabaseError, ValueError) as exc:
        if connection is not None:
            connection.close()
        if isinstance(exc, CatalogError):
            raise
        raise CatalogError(f"Не удалось открыть индекс SQLite: {exc}") from exc


def validate(connection):
    schema_version(connection)
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise CatalogError("Проверка целостности SQLite не пройдена")
    info = metadata(connection)
    if not all(info.get(key) for key in ("signature", "dimension", "root", "generation")):
        raise CatalogError("Резервная копия не содержит опубликованного индекса")
    dimension = int(info["dimension"])
    if dimension < 1:
        raise CatalogError("Некорректная размерность индекса")
    for row in connection.execute("SELECT embedding FROM images"):
        vector = np.frombuffer(row[0], dtype="<f4")
        if (
            vector.size != dimension
            or not np.isfinite(vector).all()
            or np.linalg.norm(vector) < 1e-8
        ):
            raise CatalogError("Повреждены эмбеддинги резервной копии")


def backup_catalog(settings, destination):
    destination = destination.resolve()
    if destination.exists():
        raise CatalogError("Файл резервной копии уже существует")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = connect(settings.database, readonly=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".sqlite3", delete=False
        ) as f:
            temporary = Path(f.name)
        snapshot(source, temporary)
        with closing(sqlite3.connect(temporary)) as copied:
            validate(copied)
        # Refuse to overwrite a backup created concurrently.
        os.link(temporary, destination)
    finally:
        source.close()
        if temporary:
            temporary.unlink(missing_ok=True)
    return destination


def restore_catalog(settings, source):
    """Restore only with the server stopped; keep a copy of the previous database."""
    settings.storage.mkdir(parents=True, exist_ok=True)
    if source.resolve() == settings.database.resolve():
        raise CatalogError("Источник восстановления совпадает с рабочим индексом")
    with (
        FileLock(str(settings.storage / "server.lock"), timeout=0),
        FileLock(str(settings.storage / "index.lock"), timeout=0),
    ):
        original = connect(source, readonly=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=settings.storage, suffix=".sqlite3", delete=False
            ) as f:
                temporary = Path(f.name)
            snapshot(original, temporary)
            with closing(sqlite3.connect(temporary)) as copied:
                validate(copied)
                if metadata(copied)["root"] != str(settings.catalog.resolve()):
                    raise CatalogError("Резервная копия относится к другой папке каталога")
            # Migrate the copy, never the user's backup.
            migrated = connect(temporary)
            try:
                migrated.execute("DELETE FROM staged_images")
                migrated.execute("DELETE FROM staging_context")
                migrated.execute("DELETE FROM file_state")
                migrated.execute("DELETE FROM metadata WHERE key='last_full_scan'")
                migrated.execute(
                    "UPDATE metadata SET value=? WHERE key='generation'", (uuid.uuid4().hex,)
                )
                migrated.commit()
                migrated.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                migrated.execute("PRAGMA journal_mode=DELETE")
            finally:
                migrated.close()
            if settings.database.exists():
                # Preserve raw files too: even a corrupt database can be recovered later.
                import shutil

                safety = settings.storage / f"before-restore-{uuid.uuid4().hex}"
                safety.mkdir()
                for suffix in ("", "-wal", "-shm"):
                    path = Path(str(settings.database) + suffix)
                    if path.exists():
                        shutil.copy2(path, safety / path.name)
            for suffix in ("-wal", "-shm"):
                Path(str(settings.database) + suffix).unlink(missing_ok=True)
            os.replace(temporary, settings.database)
        except (sqlite3.DatabaseError, ValueError) as exc:
            raise CatalogError(f"Восстановление не выполнено: {exc}") from exc
        finally:
            original.close()
            if temporary:
                temporary.unlink(missing_ok=True)
                for suffix in ("-wal", "-shm"):
                    Path(str(temporary) + suffix).unlink(missing_ok=True)
