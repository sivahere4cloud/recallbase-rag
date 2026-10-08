import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

NOISY_LOGGERS = (
    "httpx",
    "httpx2",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "urllib3",
)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=True,
    )

    if level == "DEBUG":
        return

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)