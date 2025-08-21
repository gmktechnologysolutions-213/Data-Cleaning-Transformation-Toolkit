import logging, sys, os

def get_logger(name="cleanit", level=os.getenv("LOG_LEVEL","INFO")):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger
