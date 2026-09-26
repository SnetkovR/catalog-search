"""Single-process local API with bounded uploads and serialized CPU inference."""

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from filelock import FileLock
from starlette.concurrency import run_in_threadpool

from .catalog import CatalogError, SearchIndex, catalog_status, connect, metadata
from .config import MAX_IMAGE_BYTES, Settings
from .images import Crop, ImageError, decode_image
from .jobs import IndexingJob, PriorityEncoder


class PayloadTooLarge(HTTPException):
    def __init__(self):
        super().__init__(413, "Файл больше 20 МБ")


class BodyLimitMiddleware:
    """Bound even chunked multipart bodies before they fill a temporary file."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        maximum = MAX_IMAGE_BYTES + 64 * 1024
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > maximum:
                raise PayloadTooLarge
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except PayloadTooLarge:
            response = JSONResponse({"detail": "Файл больше 20 МБ"}, status_code=413)
            await response(scope, receive, send)


class SearchService:
    def __init__(self, settings, encoder):
        self.settings = settings
        self.encoder = encoder
        self.gate = threading.BoundedSemaphore(1)
        self.index = None

    def refresh(self):
        status = catalog_status(self.settings)
        if not status["ready"]:
            raise CatalogError(
                "Каталог пуст. Дождитесь индексации или выполните catalog-search index"
            )
        if self.index is None or self.index.generation != status["generation"]:
            self.index = SearchIndex(self.settings, self.encoder.signature)
        return self.index

    def search(self, data, crop, top_k, exclude_identical):
        if not self.gate.acquire(blocking=False):
            raise HTTPException(503, "Сервис занят обработкой фотографии. Повторите запрос")
        try:
            started = time.perf_counter()
            index = self.refresh()
            image = decode_image(data, crop)
            encode_started = time.perf_counter()
            vector = self.encoder.encode([image])[0]
            encode_ms = (time.perf_counter() - encode_started) * 1000
            results = index.search(
                vector,
                top_k=top_k,
                exclude_hash=hashlib.sha256(data).hexdigest() if exclude_identical else None,
            )
            return {
                "results": results,
                "count": len(index.items),
                "embedding_ms": round(encode_ms, 1),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        finally:
            self.gate.release()

    def neighbors(self, image_id, top_k):
        if not self.gate.acquire(blocking=False):
            raise HTTPException(503, "Сервис занят. Повторите запрос")
        try:
            started = time.perf_counter()
            index = self.refresh()
            position = next(
                (i for i, item in enumerate(index.items) if item["id"] == image_id), None
            )
            if position is None:
                raise HTTPException(404, "Фотография не найдена")
            vector = index.vectors[position]
            results = index.search(
                vector, top_k=top_k, exclude_hash=index.items[position]["content_hash"]
            )
            return {
                "results": results,
                "count": len(index.items),
                "embedding_ms": 0,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        finally:
            self.gate.release()


def create_app(settings: Settings | None = None, *, encoder=None) -> FastAPI:
    settings = settings or Settings()

    def attach_encoder(app, instance):
        shared = PriorityEncoder(instance)
        app.state.service = SearchService(settings, shared)
        app.state.indexing_job = IndexingJob(settings, shared)
        app.state.indexing_job.start()
        app.state.model_state = "ready"

    async def initialize(app):
        try:
            from .encoder import DinoEncoder

            instance = await run_in_threadpool(DinoEncoder, settings)
            if not app.state.stopping:
                attach_encoder(app, instance)
        except Exception as exc:
            app.state.model_state = "error"
            app.state.startup_error = str(exc)
            logging.getLogger(__name__).exception("Model initialization failed")

    @asynccontextmanager
    async def lifespan(app):
        settings.storage.mkdir(parents=True, exist_ok=True)
        with FileLock(str(settings.storage / "server.lock"), timeout=0):
            app.state.service = None
            app.state.indexing_job = None
            app.state.model_state = "loading"
            app.state.startup_error = None
            app.state.stopping = False
            task = None
            if encoder is not None:
                attach_encoder(app, encoder)
            else:
                task = asyncio.create_task(initialize(app))
            try:
                yield
            finally:
                app.state.stopping = True
                if task is not None:
                    await task
                if app.state.indexing_job is not None:
                    await run_in_threadpool(app.state.indexing_job.stop)

    app = FastAPI(title="Поиск фотографий", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware)

    @app.exception_handler(CatalogError)
    async def catalog_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ImageError)
    async def image_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/api/status")
    def status():
        error = app.state.startup_error
        try:
            state = catalog_status(settings)
        except (CatalogError, sqlite3.DatabaseError) as exc:
            state = {"ready": False, "count": 0, "generation": None}
            error = str(exc)
        service = app.state.service
        compatible = service is not None and state.get("signature") == service.encoder.signature
        ready = state["ready"] and compatible and not app.state.stopping
        job = app.state.indexing_job.snapshot() if app.state.indexing_job else None
        if app.state.stopping:
            phase = "stopping"
        elif app.state.model_state == "loading":
            phase = "loading_model"
        elif app.state.model_state == "error":
            phase = "model_error"
        elif error:
            phase = "index_error"
        elif ready:
            phase = "ready"
        elif state.get("signature") and not compatible:
            phase = "incompatible_index"
        elif job and job["state"] == "running":
            phase = "indexing"
        elif job and job["state"] == "error":
            phase = "index_error"
            error = job["error"]
        else:
            phase = "empty_catalog"
        return {
            **state,
            "ready": bool(ready),
            "compatible": compatible,
            "phase": phase,
            "model": app.state.model_state,
            "error": error,
            "device": "cpu",
            "threads": settings.threads,
            "indexing": job,
        }

    @app.get("/health/live", include_in_schema=False)
    def live():
        return {"alive": True}

    @app.get("/health/ready", include_in_schema=False)
    def readiness():
        state = status()
        return JSONResponse(
            {"ready": state["ready"], "phase": state["phase"]},
            status_code=200 if state["ready"] else 503,
        )

    def search_service():
        if app.state.service is None or app.state.stopping:
            raise HTTPException(503, "Модель еще не готова к поиску", headers={"Retry-After": "3"})
        return app.state.service

    @app.get("/api/catalog")
    def catalog(limit: int = Query(24, ge=1, le=100), offset: int = Query(0, ge=0)):
        state = catalog_status(settings)
        if not state["ready"]:
            return {"items": [], "total": 0, "generation": state["generation"]}
        connection = connect(settings.database, readonly=True)
        try:
            connection.execute("BEGIN")
            rows = connection.execute(
                "SELECT id, path FROM images ORDER BY path LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            total = connection.execute("SELECT count(*) FROM images").fetchone()[0]
            return {
                "generation": metadata(connection).get("generation"),
                "items": [
                    {
                        "id": row["id"],
                        "path": row["path"],
                        "thumbnail_url": f"/api/images/{row['id']}/thumbnail",
                    }
                    for row in rows
                ],
                "total": total,
            }
        finally:
            connection.close()

    @app.get("/api/images/{image_id}/thumbnail")
    def thumbnail(image_id: str):
        if not re.fullmatch(r"[a-f0-9]{64}", image_id) or not catalog_status(settings)["ready"]:
            raise HTTPException(404, "Фотография не найдена")
        connection = connect(settings.database, readonly=True)
        try:
            row = connection.execute(
                "SELECT thumbnail FROM images WHERE id = ?", (image_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise HTTPException(404, "Фотография не найдена")
        return Response(row[0], media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    @app.post("/api/search")
    async def search(
        file: Annotated[UploadFile, File()],
        top_k: int = Form(10, ge=1, le=100),
        exclude_identical: bool = Form(False),
        crop: str | None = Form(None),
    ):
        try:
            data = await file.read(MAX_IMAGE_BYTES + 1)
        finally:
            await file.close()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Файл больше 20 МБ")
        selected_crop = None
        if crop:
            try:
                coordinates = json.loads(crop)
                if not isinstance(coordinates, list) or len(coordinates) != 4:
                    raise ValueError
                selected_crop = Crop(*(float(value) for value in coordinates))
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, "Некорректная область поиска") from exc
        return await run_in_threadpool(
            search_service().search, data, selected_crop, top_k, exclude_identical
        )

    @app.post("/api/search/catalog/{image_id}")
    def neighbors(image_id: str, top_k: int = Query(10, ge=1, le=100)):
        return search_service().neighbors(image_id, top_k)

    static = Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=static), name="static")

        @app.get("/", include_in_schema=False)
        def home():
            return FileResponse(static / "index.html")

    return app
