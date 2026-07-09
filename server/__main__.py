"""Run the Chitragupt server with `python -m server`."""

import uvicorn
from .config import settings

if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
