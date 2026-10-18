"""Append verified managed images in final-result and image-reference order."""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path
from sqlite3 import Error as DatabaseError

from core.types import RetrievalResult
from observability.logger import get_logger


class MultimodalAssembler:
    def __init__(
        self, image_root: str | Path = "data/images", *,
        image_index_db_path: str | Path = "data/db/image_index.db",
        path_resolver: Callable[[str], Path | None] | None = None,
    ) -> None:
        self.image_root = Path(image_root).expanduser().resolve()
        self.image_index_db_path = Path(image_index_db_path).expanduser().resolve()
        self._resolver = path_resolver
        self._storage = None

    def assemble(self, results: Sequence[RetrievalResult]) -> list[dict[str, str]]:
        """Return image blocks only; missing/invalid images leave text usable.

        References select images, not every image in metadata. Stored paths are
        preferred; ID-only references can use the existing ImageStorage index.
        Neither files nor an image index are created for text-only results.
        """
        from PIL import Image

        content = []
        seen = set()
        for result in results:
            refs = result.metadata.get("image_refs", [])
            if not isinstance(refs, list):
                continue
            images = result.metadata.get("images", [])
            by_id = {entry.get("id"): entry for entry in images
                     if isinstance(entry, dict) and isinstance(entry.get("id"), str)} if isinstance(images, list) else {}
            for image_id in refs:
                if not isinstance(image_id, str) or not image_id.strip() or image_id in seen:
                    continue
                try:
                    path = self._resolve(image_id, by_id.get(image_id, {}).get("path"))
                    if path is None or not path.is_relative_to(self.image_root) or not path.is_file():
                        continue
                    data = path.read_bytes()
                    # Detect the payload format, independent of file extension.
                    with Image.open(BytesIO(data)) as image:
                        mime = Image.MIME.get(image.format)
                        image.verify()
                    if mime is None or not mime.startswith("image/"):
                        continue
                    content.append({"type": "image", "mimeType": mime,
                                    "data": base64.b64encode(data).decode("ascii")})
                    seen.add(image_id)
                except (OSError, ValueError, DatabaseError, Image.DecompressionBombError):
                    get_logger(__name__).warning("Skipping unavailable retrieval image")
        return content

    def _resolve(self, image_id: str, stored_path: object) -> Path | None:
        if isinstance(stored_path, str) and stored_path.strip():
            return Path(stored_path).expanduser().resolve()
        if self._resolver is not None:
            resolved = self._resolver(image_id)
            return Path(resolved).resolve() if resolved is not None else None
        if not self.image_index_db_path.is_file():
            return None
        if self._storage is None:
            # Reuse the storage boundary's ID validation and managed-path checks.
            from ingestion.storage.image_storage import ImageStorage

            self._storage = ImageStorage(self.image_root, self.image_index_db_path)
        return self._storage.get_path(image_id)
