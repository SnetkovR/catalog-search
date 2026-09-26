"""Reproducible warm CPU timings and leave-one-file-out retrieval examples."""

import hashlib
import importlib.metadata
import platform
import time

import numpy as np

from .catalog import CatalogError, SearchIndex
from .config import MAX_IMAGE_BYTES
from .images import decode_image


def benchmark(settings, encoder, *, limit=20, repeats=3):
    if limit < 1 or repeats < 1:
        raise CatalogError("limit и repeats должны быть положительными")
    index = SearchIndex(settings, encoder.signature)
    samples = index.items[:limit]
    if not samples:
        raise CatalogError("Для замера нужен непустой индекс")
    # Fail instead of silently benchmarking a different catalog directory.
    from .catalog import connect, metadata

    connection = connect(settings.database, readonly=True)
    try:
        if metadata(connection)["root"] != str(settings.catalog.resolve()):
            raise CatalogError("Укажите --catalog, соответствующий индексу")
    finally:
        connection.close()
    prepared = []
    for item in samples:
        with (settings.catalog / item["path"]).open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["content_hash"]:
            raise CatalogError("Каталог изменился. Сначала выполните index")
        prepared.append((item, data))
    encoder.encode([decode_image(prepared[0][1])])  # Untimed warm-up.
    timings = {"decode_ms": [], "embedding_ms": [], "search_ms": [], "total_ms": []}
    examples = []
    for repeat in range(repeats):
        for item, data in prepared:
            start = time.perf_counter()
            image = decode_image(data)
            decoded = time.perf_counter()
            vector = encoder.encode([image])[0]
            encoded = time.perf_counter()
            results = index.search(vector, top_k=3, exclude_hash=item["content_hash"])
            finished = time.perf_counter()
            timings["decode_ms"].append((decoded - start) * 1000)
            timings["embedding_ms"].append((encoded - decoded) * 1000)
            timings["search_ms"].append((finished - encoded) * 1000)
            timings["total_ms"].append((finished - start) * 1000)
            if repeat == 0:
                examples.append({"query": item["path"], "results": results})
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() != "Darwin":
            peak *= 1024
        peak_mib = round(peak / 1024**2, 1)
    except ImportError:  # Windows has no standard-library resource module.
        peak_mib = None
    return {
        "device": "cpu",
        "platform": platform.platform(),
        "threads": settings.threads,
        "signature": encoder.signature,
        "catalog_size": len(index.items),
        "queries": len(prepared),
        "repeats": repeats,
        "peak_process_memory_mib": peak_mib,
        "versions": {
            name: importlib.metadata.version(name) for name in ["torch", "transformers", "numpy"]
        },
        "warm_timings": {
            key: {
                "p50": round(float(np.median(values)), 2),
                "p95": round(float(np.percentile(values, 95)), 2),
            }
            for key, values in timings.items()
        },
        "examples": examples,
        "note": "Exact file copies excluded. Warm timings exclude model load, file I/O and HTTP. "
        "These examples are not a labeled quality benchmark.",
    }
