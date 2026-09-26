"""Configuration shared by indexing, CLI and web search."""

import os
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "main"
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

    def __post_init__(self):
        if self.threads < 1:
            raise ValueError("Число потоков должно быть положительным")

    @property
    def database(self) -> Path:
        return self.storage / "catalog.sqlite3"
