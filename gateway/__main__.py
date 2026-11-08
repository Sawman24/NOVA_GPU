"""Start the Gateway:  python -m gateway"""
import logging

import uvicorn

from nova_common.config import load_config

from .server import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    g = cfg.get("gateway") or {}
    uvicorn.run(create_app(cfg), host=g.get("host", "0.0.0.0"), port=int(g.get("port", 8800)), log_level="info")


if __name__ == "__main__":
    main()
