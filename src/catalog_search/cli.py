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
    index.add_argument("--verify", action="store_true", help="Прочитать и сверить хеши всех файлов")
    backup = commands.add_parser("backup", help="Создать проверенную резервную копию индекса")
    backup.add_argument("destination", type=Path)
    restore = commands.add_parser("restore", help="Восстановить индекс при остановленном сервере")
    restore.add_argument("source", type=Path)
    search = commands.add_parser("search", help="Найти похожие изображения")
    search.add_argument("image", type=Path)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--exclude-identical", action="store_true")
    search.add_argument("--crop", type=float, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    commands.add_parser("status", help="Состояние индекса без загрузки модели")
    serve = commands.add_parser("serve", help="Запустить веб-интерфейс")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--index-interval",
        type=float,
        default=60,
        help="Пауза между фоновыми проверками в секундах; 0 отключает задачу",
    )
    serve.add_argument("--index-batch-size", type=int, default=4)
    serve.add_argument(
        "--index-verify-interval",
        type=float,
        default=86400,
        help="Интервал полной сверки хешей; 0 проверяет каждый проход",
    )
    serve.add_argument(
        "--index-settle-seconds",
        type=float,
        default=2,
        help="Не индексировать файлы, измененные менее N секунд назад",
    )
    bench = commands.add_parser("benchmark", help="Замерить CPU и поиск без самосовпадений")
    bench.add_argument("--limit", type=int, default=20)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--output", type=Path)
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
            index_interval=getattr(args, "index_interval", 60),
            index_batch_size=getattr(args, "index_batch_size", 4),
            index_settle_seconds=getattr(args, "index_settle_seconds", 2),
            index_verify_interval=getattr(args, "index_verify_interval", 86400),
        )
        run(args, settings)
    except (ValueError, OSError, RuntimeError, Timeout) as exc:
        message = (
            "Индекс занят: остановите сервер или дождитесь индексации"
            if isinstance(exc, Timeout)
            else str(exc)
        )
        print(f"Ошибка: {message}", file=sys.stderr)
        raise SystemExit(1) from exc


def run(args, settings):
    if args.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)
        return

    from .catalog import SearchIndex, catalog_status, index_catalog, report_json

    if args.command in {"backup", "restore"}:
        from .storage import backup_catalog, restore_catalog

        if args.command == "backup":
            print(backup_catalog(settings, args.destination))
        else:
            restore_catalog(settings, args.source)
            print("Индекс восстановлен")
        return

    if args.command == "status":
        print(json.dumps(catalog_status(settings), ensure_ascii=False, indent=2))
        return

    from .config import MAX_IMAGE_BYTES
    from .encoder import DinoEncoder
    from .images import Crop, decode_image

    print("Загрузка DINOv2 Small на CPU…", file=sys.stderr)
    load_started = time.perf_counter()
    encoder = DinoEncoder(settings)
    load_seconds = time.perf_counter() - load_started
    if args.command == "benchmark":
        from .benchmark import benchmark

        result = benchmark(settings, encoder, limit=args.limit, repeats=args.repeats)
        result["model_load_seconds"] = round(load_seconds, 3)
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        print(output)
        return
    if args.command == "index":
        report = index_catalog(
            settings, encoder, batch_size=args.batch_size, rebuild=args.rebuild, verify=args.verify
        )
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
