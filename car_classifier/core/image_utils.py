import base64
from typing import Tuple

from fastapi import UploadFile


async def convert_to_base64(file: UploadFile) -> Tuple[str, str]:
    content = await file.read()
    if not content:
        raise ValueError("Uploaded file is empty")
    mime_type = file.content_type or "application/octet-stream"
    return base64.b64encode(content).decode("utf-8"), mime_type
