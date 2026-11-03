import logging

from bitcoin_aml.utils.logging import configure_logging, get_logger


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("bitcoin_aml.some.module")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "bitcoin_aml.some.module"


def test_configure_logging_installs_a_handler() -> None:
    configure_logging()

    root = logging.getLogger()
    assert len(root.handlers) >= 1


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    root = logging.getLogger()
    handler_count_after_first_call = len(root.handlers)

    configure_logging()

    assert len(root.handlers) == handler_count_after_first_call
