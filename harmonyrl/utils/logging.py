import logging, sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger