"""Local command line entry point; no background infrastructure required."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from filelock import Timeout

from .config import Settings


def build_parser():
    parser = argparse.ArgumentParser(description="Поиск фотографий на CPU")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog"))
    parser.add_argument("--storage", type=Path, default=Path("var"))
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--threads", type=int, default=Settings().threads)
    parser.add_argument("--offline", action="store_true", help="Использовать только скачанные веса")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="Создать или обновить индекс")
    index.add_argument("--batch-size", type=int, default=4)
    index.add_argument("--rebuild", action="store_true")
    search = commands.add_parser("search", help="Найти похожие изображения")
    search.add_argument("image", type=Path)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--exclude-identical", action="store_true")
    search.add_argument("--crop", type=float, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    commands.add_parser("status", help="Состояние индекса без загрузки модели")
    serve = commands.add_parser("serve", help="Запустить веб-интерфейс")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        settings = Settings(
            catalog=args.catalog,
            storage=args.storage,
            model_cache=args.model_cache,
            threads=args.threads,
            offline=args.offline,
        )
        run(args, settings)
    except (ValueError, OSError, RuntimeError, Timeout) as exc:
        message = "Индексация уже запущена" if isinstance(exc, Timeout) else str(exc)
        print(f"Ошибка: {message}", file=sys.stderr)
        raise SystemExit(1) from exc


def run(args, settings):
    if args.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)
        return

    from .catalog import SearchIndex, catalog_status, index_catalog, report_json

    if args.command == "status":
        print(json.dumps(catalog_status(settings), ensure_ascii=False, indent=2))
        return

    from .config import MAX_IMAGE_BYTES
    from .encoder import DinoEncoder
    from .images import Crop, decode_image

    print("Загрузка DINOv2 Small на CPU…", file=sys.stderr)
    encoder = DinoEncoder(settings)
    if args.command == "index":
        report = index_catalog(settings, encoder, batch_size=args.batch_size, rebuild=args.rebuild)
        print(report_json(report))
        return
    started = time.perf_counter()
    with args.image.open("rb") as stream:
        data = stream.read(MAX_IMAGE_BYTES + 1)
    crop = Crop(*args.crop) if args.crop else None
    image = decode_image(data, crop)
    vector = encoder.encode([image])[0]
    index = SearchIndex(settings, encoder.signature)
    results = index.search(
        vector,
        top_k=args.top_k,
        exclude_hash=hashlib.sha256(data).hexdigest() if args.exclude_identical else None,
    )
    print(
        json.dumps(
            {"results": results, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
