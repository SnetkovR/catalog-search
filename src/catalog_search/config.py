"""Configuration shared by indexing, CLI and web search."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
PREPROCESS_VERSION = "rgb-exif-letterbox224-imagenet-cls-l2-v1"
DIMENSION = 384
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000


@dataclass(frozen=True)
class Settings:
    catalog: Path = Path("data/catalog")
    storage: Path = Path("var")
    model_cache: Path = Path(".cache/huggingface")
    threads: int = min(4, os.cpu_count() or 1)
    offline: bool = False
    index_interval: float = 10
    index_batch_size: int = 4
    index_settle_seconds: float = 2

    def __post_init__(self):
        if self.threads < 1:
            raise ValueError("Число потоков должно быть положительным")
        if not math.isfinite(self.index_interval) or self.index_interval < 0:
            raise ValueError("Интервал индексации должен быть неотрицательным")
        if self.index_batch_size < 1:
            raise ValueError("Размер пакета индексации должен быть положительным")
        if not math.isfinite(self.index_settle_seconds) or self.index_settle_seconds < 0:
            raise ValueError("Задержка стабилизации файлов должна быть неотрицательной")

    @property
    def database(self) -> Path:
        return self.storage / "catalog.sqlite3"
