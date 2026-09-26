import sqlite3

import pytest
from filelock import FileLock, Timeout
from test_catalog import catalog  # noqa: F401

from catalog_search.catalog import SearchIndex, catalog_status, index_catalog
from catalog_search.storage import (
    SCHEMA_VERSION,
    CatalogError,
    backup_catalog,
    connect,
    restore_catalog,
)


def test_migration_preserves_legacy_vectors_and_keeps_backup(catalog):  # noqa: F811
    settings, encoder = catalog
    index_catalog(settings, encoder)
    before = catalog_status(settings)
    with sqlite3.connect(settings.database) as db:
        for table in ("file_state", "staged_images", "staging_context"):
            db.execute(f"DROP TABLE {table}")
        db.execute("UPDATE metadata SET value='1' WHERE key='schema_version'")
        db.execute("PRAGMA user_version=0")
    index_catalog(settings, encoder)
    assert encoder.encoded == 2
    assert catalog_status(settings) == before
    copies = list(settings.storage.glob("*.before-v2-*.sqlite3"))
    assert len(copies) == 1
    with sqlite3.connect(copies[0]) as db:
        assert (
            db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0] == "1"
        )
    with connect(settings.database) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_future_schema_is_rejected_without_modification(catalog):  # noqa: F811
    settings, encoder = catalog
    index_catalog(settings, encoder)
    with sqlite3.connect(settings.database) as db:
        db.execute("PRAGMA user_version=999")
        db.execute("UPDATE metadata SET value='999' WHERE key='schema_version'")
    for readonly in (True, False):
        with pytest.raises(CatalogError, match="Неподдерживаемая"):
            connect(settings.database, readonly=readonly)
    with sqlite3.connect(settings.database) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 999


def test_backup_restore_and_corrupt_database_recovery(catalog, tmp_path):  # noqa: F811
    settings, encoder = catalog
    index_catalog(settings, encoder)
    backup = backup_catalog(settings, tmp_path / "backup.sqlite3")
    with pytest.raises(CatalogError, match="уже существует"):
        backup_catalog(settings, backup)
    (settings.catalog / "red.png").unlink()
    index_catalog(settings, encoder)
    restore_catalog(settings, backup)
    assert len(SearchIndex(settings, encoder.signature).items) == 2
    settings.database.write_bytes(b"broken SQLite")
    restore_catalog(settings, backup)
    assert len(SearchIndex(settings, encoder.signature).items) == 2
    assert len(list(settings.storage.glob("before-restore-*"))) == 2


def test_restore_rejects_bad_backup_and_running_server(catalog, tmp_path):  # noqa: F811
    settings, encoder = catalog
    index_catalog(settings, encoder)
    before = catalog_status(settings)
    bad = tmp_path / "bad.sqlite3"
    bad.write_bytes(b"bad")
    with pytest.raises(CatalogError):
        restore_catalog(settings, bad)
    backup = backup_catalog(settings, tmp_path / "backup.sqlite3")
    with FileLock(str(settings.storage / "server.lock")):
        with pytest.raises(Timeout):
            restore_catalog(settings, backup)
    assert catalog_status(settings) == before


def test_backup_includes_committed_wal_pages(catalog, tmp_path):  # noqa: F811
    settings, encoder = catalog
    index_catalog(settings, encoder)
    connection = connect(settings.database)
    try:
        connection.execute("UPDATE metadata SET value='wal-generation' WHERE key='generation'")
        connection.commit()
        backup = backup_catalog(settings, tmp_path / "backup.sqlite3")
        with sqlite3.connect(backup) as db:
            assert (
                db.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()[0]
                == "wal-generation"
            )
    finally:
        connection.close()
