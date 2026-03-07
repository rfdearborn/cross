import logging

import uvicorn

from cross.config import settings


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(
        "cross.proxy:app",
        host=settings.listen_host,
        port=settings.listen_port,
        log_level="warning",  # Quiet uvicorn's own logs
    )


if __name__ == "__main__":
    main()
