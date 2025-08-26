import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application wide logging.

    Parameters
    ----------
    level: int
        The logging level to configure the root logger with.
    """
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
