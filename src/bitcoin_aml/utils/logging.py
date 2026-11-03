# consistent logging configuration and logger factory, so run/model/
# selection state is observable the same way from every entry point

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """
    configures the root logging handler and format once per process

    safe to call multiple times (e.g. once per script entry point, once
    per test run): later calls are a no-op so nothing double-configures
    or clobbers a caller's own handler setup
    """
    global _configured

    if _configured:
        return

    logging.basicConfig(level=level, format=_LOG_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    returns a module-scoped logger

    does not configure handlers itself -- call configure_logging() once
    at the application entry point (a script or a test fixture); modules
    that only need to emit log records should call this and nothing else
    """
    return logging.getLogger(name)
