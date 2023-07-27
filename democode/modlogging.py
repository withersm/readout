import logging

logger = logging.getLogger(__name__)


def someFunction():
    log = logger.getChild(__name__)
    log.info("Hello World")
    return True
