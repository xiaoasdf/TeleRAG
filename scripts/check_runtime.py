from __future__ import annotations

import json

from src.api.service import TeleRAGService
from src.runtime import get_compute_device, get_faiss_gpu_status


def main() -> None:
    service = TeleRAGService()
    supported, reason = get_faiss_gpu_status()
    payload = {
        "device": get_compute_device(),
        "faiss_gpu_supported": supported,
        "faiss_gpu_status": reason,
        "health": service.health(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
