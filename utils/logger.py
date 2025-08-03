import logging
import sys

def setup_logging():
    logger = logging.getLogger("mcp_server")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S'
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger