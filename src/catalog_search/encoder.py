"""A single reusable DINOv2 encoder, explicitly restricted to CPU."""

import threading
from collections.abc import Sequence

import numpy as np
from PIL import Image

from .config import DIMENSION, MODEL_ID, MODEL_REVISION, PREPROCESS_VERSION, Settings
from .images import prepare_tensor


class DinoEncoder:
    dimension = DIMENSION

    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoModel

        torch.set_num_threads(settings.threads)
        self._torch = torch
        self._lock = threading.Lock()
        self.model = (
            AutoModel.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(settings.model_cache),
                local_files_only=settings.offline,
                use_safetensors=True,
            )
            .to(device="cpu", dtype=torch.float32)
            .eval()
        )
        revision = getattr(self.model.config, "_commit_hash", None) or MODEL_REVISION
        self.signature = f"{MODEL_ID}@{revision}:{PREPROCESS_VERSION}"

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        pixels = np.stack([prepare_tensor(image) for image in images])
        with self._lock, self._torch.inference_mode():
            tensor = self._torch.from_numpy(pixels).to("cpu")
            features = self.model(pixel_values=tensor).last_hidden_state[:, 0, :]
            features = self._torch.nn.functional.normalize(features, dim=1)
            result = features.cpu().numpy().astype(np.float32, copy=True)
        if result.shape != (len(images), self.dimension) or not np.isfinite(result).all():
            raise RuntimeError("Модель вернула некорректные эмбеддинги")
        return result
