import logging
import logging.handlers
import os
import sys

def get_logger(name="quant", level="INFO", log_dir=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')

    if sys.platform.startswith('win'):
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except:
            pass

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.TimedRotatingFileHandler(
            os.path.join(log_dir, "quant.log"), when='midnight', interval=1, backupCount=30, encoding='utf-8'
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        efh = logging.handlers.TimedRotatingFileHandler(
            os.path.join(log_dir, "quant_error.log"), when='midnight', interval=1, backupCount=7, encoding='utf-8'
        )
        efh.setLevel(logging.ERROR)
        efh.setFormatter(formatter)
        logger.addHandler(efh)

    return logger
