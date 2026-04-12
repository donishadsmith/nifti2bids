import logging

import rich

from bidsaid.logging import setup_logger, _add_default_handler


def test_setup_logger(caplog):
    """Test for ``setup_logger``"""
    logger = setup_logger("test")
    assert isinstance(logger, logging.Logger)

    with caplog.at_level(logging.INFO):
        logger.info("TEST")

    assert "TEST" in caplog.text

    # Root has handler in pytest env
    logger.handlers.clear()
    logger = setup_logger("test")
    logger = _add_default_handler(logger)
    assert isinstance(logger.handlers[0], rich.logging.RichHandler)

    # Clear root handler
    logging.getLogger().handlers.clear()
    logger = setup_logger("test2")
    assert isinstance(logger.handlers[0], rich.logging.RichHandler)

    logger = setup_logger("test2", 20)
    assert logger.level == 20
